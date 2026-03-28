use plonkish_backend::util::binary_extension_fields::B128;
use plonkish_backend::util::blaze_transcript::BlazeBlake2sTranscript;
use plonkish_backend::util::transcript::{
    Blake2sTranscript, InMemoryTranscript, TranscriptRead, TranscriptWrite,
};
use itertools::Itertools;
use rayon::prelude::*;
use rand::rngs::OsRng;
use plonkish_backend::{
    pcs::multilinear::blaze::{
        self, setup as blaze_setup, trim as blaze_trim, BlazeCommitment,
        CommitmentChunk as BlazeCommitmentChunk,
    },
    util::{
        arithmetic::Field,
        avx_int_types::{u64::Blazeu64, BlazeField},
        hash::{Blake2s, Hash},
    },
};

use std::{
    env::args,
    fmt::Display,
    fs::{create_dir, File, OpenOptions},
    io::Write,
    ops::Range,
    path::Path,
    time::{Duration, Instant},
};

const OUTPUT_DIR: &str = "./bench_data/pcs";

fn main() {
    let (systems, log_per_row_range, blaze_log_rows) = parse_args();
    create_output(&systems);
    log_per_row_range
        .for_each(|log_per_row| systems.iter().for_each(|system| system.bench(log_per_row, blaze_log_rows)));
}

fn average_duration(samples: &[Duration]) -> Duration {
    let window = if samples.len() > 2 {
        &samples[2..]
    } else {
        samples
    };
    let sum = window.iter().copied().sum::<Duration>();
    sum / window.len().max(1) as u32
}

fn print_bench_summary(
    pcs: System,
    log_per_row: usize,
    log_rows: usize,
    num_rows: usize,
    poly_size: usize,
    queries: usize,
    sample_size: usize,
    raa_repeat_interleave: Duration,
    raa_first_accumulate: Duration,
    raa_second_interleave: Duration,
    raa_second_accumulate: Duration,
    raa_third_interleave: Duration,
    raa_third_accumulate: Duration,
    raa_total: Duration,
    merkle: Duration,
    transcript_write: Duration,
    commit: Duration,
    prove: Duration,
    verify: Duration,
) {
    println!(
        "pcs={} log_per_row={} log_rows={} num_rows={} poly_size={} queries={} sample_size={} raa_repeat_interleave_ms={} raa_first_accumulate_ms={} raa_second_interleave_ms={} raa_second_accumulate_ms={} raa_third_interleave_ms={} raa_third_accumulate_ms={} raa_total_ms={} merkle_ms={} transcript_write_ms={} commit_ms={} prove_ms={} verify_ms={}",
        pcs,
        log_per_row,
        log_rows,
        num_rows,
        poly_size,
        queries,
        sample_size,
        raa_repeat_interleave.as_millis(),
        raa_first_accumulate.as_millis(),
        raa_second_interleave.as_millis(),
        raa_second_accumulate.as_millis(),
        raa_third_interleave.as_millis(),
        raa_third_accumulate.as_millis(),
        raa_total.as_millis(),
        merkle.as_millis(),
        transcript_write.as_millis(),
        commit.as_millis(),
        prove.as_millis(),
        verify.as_millis()
    );
}

fn bench_blaze<F, H, T1, T2>(log_per_row: usize, pcs: System, log_rows: usize, queries: usize)
where
    F: BlazeField,
    H: Hash,
    T1: TranscriptRead<BlazeCommitmentChunk<H>, B128>
        + TranscriptWrite<BlazeCommitmentChunk<H>, B128>
        + InMemoryTranscript<Param = ()>,
    T2: TranscriptRead<BlazeCommitmentChunk<H>, F>
        + TranscriptWrite<BlazeCommitmentChunk<H>, F>
        + InMemoryTranscript<Param = ()>,
{
    let mut b128_transcript = T1::new(());
    let mut blaze_transcript = T2::new(());

    let mut rng = OsRng;
    let num_rows = 1 << log_rows;
    let poly_size = 1 << log_per_row;
    let param = blaze_setup::<H>(poly_size, 2, &mut rng, Some(num_rows), Some(queries));
    let (pp, vp) = blaze_trim::<H>(&param, poly_size, 1);

    let matrix = (0..num_rows)
        .into_par_iter()
        .map(|_| F::rand_vec(poly_size))
        .collect::<Vec<_>>();

    let sample_size = sample_size(log_per_row);
    println!(
        "bench_start pcs={} log_per_row={} log_rows={} num_rows={} poly_size={} queries={} sample_size={}",
        pcs,
        log_per_row,
        log_rows,
        num_rows,
        poly_size,
        queries,
        sample_size
    );

    let mut commit_times = Vec::new();
    let mut prove_times = Vec::new();
    let mut commit_timing_samples = Vec::new();
    let mut comm = BlazeCommitment::default();
    let mut point = Vec::new();

    for sample_idx in 0..sample_size {
        let commit_start = Instant::now();
        let (new_comm, commit_timings) =
            blaze::commit_and_write_with_timings::<F, H>(&pp, &matrix, &mut blaze_transcript);
        comm = new_comm;
        let commit_elapsed = commit_start.elapsed();
        commit_times.push(commit_elapsed);
        commit_timing_samples.push(commit_timings);
        println!(
            "bench_sample pcs={} sample={} phase=commit elapsed_ms={} raa_repeat_interleave_ms={} raa_first_accumulate_ms={} raa_second_interleave_ms={} raa_second_accumulate_ms={} raa_third_interleave_ms={} raa_third_accumulate_ms={} raa_total_ms={} merkle_ms={} transcript_write_ms={}",
            pcs,
            sample_idx,
            commit_elapsed.as_millis(),
            commit_timings.raa_repeat_interleave.as_millis(),
            commit_timings.raa_first_accumulate.as_millis(),
            commit_timings.raa_second_interleave.as_millis(),
            commit_timings.raa_second_accumulate.as_millis(),
            commit_timings.raa_third_interleave.as_millis(),
            commit_timings.raa_third_accumulate.as_millis(),
            commit_timings.raa_total.as_millis(),
            commit_timings.merkle.as_millis(),
            commit_timings.transcript_write.as_millis()
        );

        point = b128_transcript.squeeze_challenges(log_per_row);
        let prove_start = Instant::now();
        blaze::open(
            &pp,
            &matrix,
            &comm,
            &point,
            &B128::ZERO,
            &mut blaze_transcript,
            &mut b128_transcript,
        )
        .unwrap();
        let prove_elapsed = prove_start.elapsed();
        prove_times.push(prove_elapsed);
        println!(
            "bench_sample pcs={} sample={} phase=prove elapsed_ms={}",
            pcs,
            sample_idx,
            prove_elapsed.as_millis()
        );
    }

    let commit_avg = average_duration(&commit_times);
    let prove_avg = average_duration(&prove_times);
    let raa_repeat_interleave_avg = average_duration(
        &commit_timing_samples
            .iter()
            .map(|t| t.raa_repeat_interleave)
            .collect::<Vec<_>>(),
    );
    let raa_first_accumulate_avg = average_duration(
        &commit_timing_samples
            .iter()
            .map(|t| t.raa_first_accumulate)
            .collect::<Vec<_>>(),
    );
    let raa_second_interleave_avg = average_duration(
        &commit_timing_samples
            .iter()
            .map(|t| t.raa_second_interleave)
            .collect::<Vec<_>>(),
    );
    let raa_second_accumulate_avg = average_duration(
        &commit_timing_samples
            .iter()
            .map(|t| t.raa_second_accumulate)
            .collect::<Vec<_>>(),
    );
    let raa_third_interleave_avg = average_duration(
        &commit_timing_samples
            .iter()
            .map(|t| t.raa_third_interleave)
            .collect::<Vec<_>>(),
    );
    let raa_third_accumulate_avg = average_duration(
        &commit_timing_samples
            .iter()
            .map(|t| t.raa_third_accumulate)
            .collect::<Vec<_>>(),
    );
    let raa_total_avg = average_duration(
        &commit_timing_samples
            .iter()
            .map(|t| t.raa_total)
            .collect::<Vec<_>>(),
    );
    let merkle_avg = average_duration(
        &commit_timing_samples
            .iter()
            .map(|t| t.merkle)
            .collect::<Vec<_>>(),
    );
    let transcript_write_avg = average_duration(
        &commit_timing_samples
            .iter()
            .map(|t| t.transcript_write)
            .collect::<Vec<_>>(),
    );

    writeln!(&mut pcs.commit_output(), "{log_per_row}, {}", commit_avg.as_millis()).unwrap();
    writeln!(&mut pcs.output(), "{log_per_row}, {}", prove_avg.as_millis()).unwrap();

    let blaze_proof = blaze_transcript.into_proof();
    let b128_proof = b128_transcript.into_proof();

    let mut verify_times = Vec::new();
    for sample_idx in 0..sample_size {
        let blaze_proof = blaze_proof.clone();
        let b128_proof = b128_proof.clone();
        let mut blaze_transcript = T2::from_proof((), blaze_proof.as_slice());
        let mut b128_transcript = T1::from_proof((), b128_proof.as_slice());
        let verify_start = Instant::now();
        let _ = blaze::verify(
            &vp,
            &comm,
            &point,
            &F::zero(),
            &mut b128_transcript,
            &mut blaze_transcript,
        );
        let verify_elapsed = verify_start.elapsed();
        verify_times.push(verify_elapsed);
        println!(
            "bench_sample pcs={} sample={} phase=verify elapsed_ms={}",
            pcs,
            sample_idx,
            verify_elapsed.as_millis()
        );
    }

    let verify_avg = average_duration(&verify_times);
    print_bench_summary(
        pcs,
        log_per_row,
        log_rows,
        num_rows,
        poly_size,
        queries,
        sample_size,
        raa_repeat_interleave_avg,
        raa_first_accumulate_avg,
        raa_second_interleave_avg,
        raa_second_accumulate_avg,
        raa_third_interleave_avg,
        raa_third_accumulate_avg,
        raa_total_avg,
        merkle_avg,
        transcript_write_avg,
        commit_avg,
        prove_avg,
        verify_avg,
    );

    writeln!(&mut pcs.verify_output(), "{:?}: {:?}", log_per_row, verify_avg.as_millis()).unwrap();
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum System {
    Blaze64,
}

impl System {
    fn all() -> Vec<System> {
        vec![System::Blaze64]
    }

    fn output_path(&self) -> String {
        format!("{OUTPUT_DIR}/open_{self}")
    }

    fn output(&self) -> File {
        OpenOptions::new().append(true).open(self.output_path()).unwrap()
    }

    fn commit_output_path(&self) -> String {
        format!("{OUTPUT_DIR}/commit_{self}")
    }

    fn size_output_path(&self) -> String {
        format!("{OUTPUT_DIR}/size_{self}")
    }

    fn verify_output_path(&self) -> String {
        format!("{OUTPUT_DIR}/verify_{self}")
    }

    fn batch_open_output_path(&self) -> String {
        format!("{OUTPUT_DIR}/batch_open_{self}")
    }

    fn batch_commit_output_path(&self) -> String {
        format!("{OUTPUT_DIR}/batch_commit_{self}")
    }

    fn commit_output(&self) -> File {
        OpenOptions::new().append(true).open(self.commit_output_path()).unwrap()
    }

    fn size_output(&self) -> File {
        OpenOptions::new().append(true).open(self.size_output_path()).unwrap()
    }

    fn verify_output(&self) -> File {
        OpenOptions::new().append(true).open(self.verify_output_path()).unwrap()
    }

    fn bench(&self, log_per_row: usize, blaze_log_rows: usize) {
        match self {
            System::Blaze64 => {
                bench_blaze::<Blazeu64, Blake2s, Blake2sTranscript<_>, BlazeBlake2sTranscript<_>>(
                    log_per_row,
                    System::Blaze64,
                    blaze_log_rows,
                    2004,
                );
            }
        }
    }
}

impl Display for System {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            System::Blaze64 => write!(f, "blaze64"),
        }
    }
}

fn parse_args() -> (Vec<System>, Range<usize>, usize) {
    let (systems, log_per_row_range, blaze_log_rows) = args()
        .chain(Some("".to_string()))
        .tuple_windows()
        .fold(
            (Vec::new(), 15..16, 5usize),
            |(mut systems, mut log_per_row_range, mut blaze_log_rows), (key, value)| {
                match key.as_str() {
                    "--system" => match value.as_str() {
                        "all" => systems = System::all(),
                        "blaze64" => systems.push(System::Blaze64),
                        _ => panic!("system should be one of {{all,blaze64}}"),
                    },
                    "--log-per-row" => {
                        if let Some((start, end)) = value.split_once("..") {
                            log_per_row_range = start
                                .parse()
                                .expect("log_per_row range start to be usize")
                                ..end.parse().expect("log_per_row range end to be usize");
                        } else {
                            log_per_row_range.start =
                                value.parse().expect("log_per_row to be usize");
                            log_per_row_range.end = log_per_row_range.start + 1;
                        }
                    }
                    "--log-rows" => {
                        blaze_log_rows = value.parse().expect("log_rows to be usize");
                    }
                    _ => {}
                }
                (systems, log_per_row_range, blaze_log_rows)
            },
        );

    let mut systems = systems.into_iter().sorted().dedup().collect_vec();
    if systems.is_empty() {
        systems = System::all();
    }
    (systems, log_per_row_range, blaze_log_rows)
}

fn create_output(systems: &[System]) {
    if !Path::new(OUTPUT_DIR).exists() {
        create_dir(OUTPUT_DIR).unwrap();
    }
    for system in systems {
        File::create(system.output_path()).unwrap();
        File::create(system.batch_open_output_path()).unwrap();
        File::create(system.commit_output_path()).unwrap();
        File::create(system.batch_commit_output_path()).unwrap();
        File::create(system.size_output_path()).unwrap();
        File::create(system.verify_output_path()).unwrap();
    }
}

fn sample_size(_log_per_row: usize) -> usize {
    1
}
