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
    commit: Duration,
    prove: Duration,
    verify: Duration,
) {
    println!(
        "pcs={} log_per_row={} log_rows={} commit_ms={} prove_ms={} verify_ms={}",
        pcs,
        log_per_row,
        log_rows,
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

    let mut commit_times = Vec::new();
    let mut prove_times = Vec::new();
    let mut comm = BlazeCommitment::default();
    let mut point = Vec::new();

    for _ in 0..sample_size {
        let commit_start = Instant::now();
        comm = blaze::commit_and_write::<F, H>(&pp, &matrix, &mut blaze_transcript);
        commit_times.push(commit_start.elapsed());

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
        prove_times.push(prove_start.elapsed());
    }

    let commit_avg = average_duration(&commit_times);
    let prove_avg = average_duration(&prove_times);

    writeln!(&mut pcs.commit_output(), "{log_per_row}, {}", commit_avg.as_millis()).unwrap();
    writeln!(&mut pcs.output(), "{log_per_row}, {}", prove_avg.as_millis()).unwrap();

    let blaze_proof = blaze_transcript.into_proof();
    let b128_proof = b128_transcript.into_proof();

    let mut verify_times = Vec::new();
    for _ in 0..sample_size {
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
        verify_times.push(verify_start.elapsed());
    }

    let verify_avg = average_duration(&verify_times);
    print_bench_summary(pcs, log_per_row, log_rows, commit_avg, prove_avg, verify_avg);

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
