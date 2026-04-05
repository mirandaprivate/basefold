use rand::rngs::OsRng;

use itertools::Itertools;
use plonkish_backend::{
    pcs::{
        multilinear::{MultilinearBrakedown},
        PolynomialCommitmentScheme,
    },
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::PrimeField,
        code::{BrakedownSpec1, BrakedownSpec3, BrakedownSpec6},
        hash::Blake2s256,
        new_fields::Mersenne127,
        transcript::{
            Blake2sTranscript, InMemoryTranscript, TranscriptRead, TranscriptWrite,
        },
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

const OUTPUT_DIR: &str = "./bench_data/brakedown";

fn main() {
    let (systems, k_range) = parse_args();
    create_output(&systems);
    k_range.for_each(|k| systems.iter().for_each(|system| system.bench(k)));
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
    k: usize,
    poly_size: usize,
    sample_size: usize,
    proof_size_bits: usize,
    commit: Duration,
    open: Duration,
    verify: Duration,
) {
    println!(
        "bench_summary pcs={} k={} poly_size={} sample_size={} proof_size_bits={}",
        pcs, k, poly_size, sample_size, proof_size_bits
    );
    println!("commit_time_ms={}", commit.as_millis());
    println!("open_time_ms={}", open.as_millis());
    println!("verify_time_ms={}", verify.as_millis());
}

fn bench_pcs<F, Pcs, T>(k: usize, pcs: System)
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
    T: TranscriptRead<Pcs::CommitmentChunk, F>
        + TranscriptWrite<Pcs::CommitmentChunk, F>
        + InMemoryTranscript<Param = ()>,
{
    let mut rng = OsRng;
    let poly_size = 1 << k;
    let param = Pcs::setup(poly_size, 1, &mut rng).unwrap();
    let (pp, vp) = Pcs::trim(&param, poly_size, 1).unwrap();
    let poly = MultilinearPolynomial::rand(k, OsRng);

    let sample_size = sample_size(k);
    println!(
        "bench_start pcs={} k={} poly_size={} sample_size={}",
        pcs, k, poly_size, sample_size
    );

    let mut commit_times = Vec::new();
    let mut open_times = Vec::new();

    for sample_idx in 0..sample_size {
        let mut transcript = T::new(());

        let commit_start = Instant::now();
        let comm = Pcs::commit_and_write(&pp, &poly, &mut transcript).unwrap();
        let commit_elapsed = commit_start.elapsed();
        commit_times.push(commit_elapsed);

        let point = transcript.squeeze_challenges(k);
        let eval = poly.evaluate(point.as_slice());
        transcript.write_field_element(&eval).unwrap();

        let open_start = Instant::now();
        Pcs::open(&pp, &poly, &comm, &point, &eval, &mut transcript).unwrap();
        let open_elapsed = open_start.elapsed();
        open_times.push(open_elapsed);

        println!(
            "bench_sample pcs={} sample={} phase=commit elapsed_ms={}",
            pcs,
            sample_idx,
            commit_elapsed.as_millis()
        );
        println!(
            "bench_sample pcs={} sample={} phase=open elapsed_ms={}",
            pcs,
            sample_idx,
            open_elapsed.as_millis()
        );
    }

    let commit_avg = average_duration(&commit_times);
    let open_avg = average_duration(&open_times);

    writeln!(&mut pcs.commit_output(), "{k}, {}", commit_avg.as_millis()).unwrap();
    writeln!(&mut pcs.output(), "{k}, {}", open_avg.as_millis()).unwrap();

    let mut transcript = T::new(());
    let comm = Pcs::commit_and_write(&pp, &poly, &mut transcript).unwrap();
    let point = transcript.squeeze_challenges(k);
    let eval = poly.evaluate(point.as_slice());
    transcript.write_field_element(&eval).unwrap();
    Pcs::open(&pp, &poly, &comm, &point, &eval, &mut transcript).unwrap();
    let proof = transcript.into_proof();
    let proof_size_bits = proof.len() * 8;

    let mut verify_times = Vec::new();
    for sample_idx in 0..sample_size {
        let mut transcript = T::from_proof((), proof.as_slice());
        let verify_start = Instant::now();
        let result = Pcs::verify(
            &vp,
            &Pcs::read_commitment(&vp, &mut transcript).unwrap(),
            &transcript.squeeze_challenges(k),
            &transcript.read_field_element().unwrap(),
            &mut transcript,
        );
        let verify_elapsed = verify_start.elapsed();
        verify_times.push(verify_elapsed);
        println!(
            "bench_sample pcs={} sample={} phase=verify elapsed_ms={}",
            pcs,
            sample_idx,
            verify_elapsed.as_millis()
        );
        assert_eq!(result, Ok(()));
    }

    let verify_avg = average_duration(&verify_times);
    writeln!(&mut pcs.verify_output(), "{k}, {}", verify_avg.as_millis()).unwrap();
    writeln!(&mut pcs.size_output(), "{k}, {proof_size_bits}").unwrap();

    print_bench_summary(
        pcs,
        k,
        poly_size,
        sample_size,
        proof_size_bits,
        commit_avg,
        open_avg,
        verify_avg,
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum System {
    Spec1,
    Spec3,
    Spec6,
}

impl System {
    fn all() -> Vec<System> {
        vec![System::Spec1, System::Spec3, System::Spec6]
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

    fn commit_output(&self) -> File {
        OpenOptions::new()
            .append(true)
            .open(self.commit_output_path())
            .unwrap()
    }

    fn size_output(&self) -> File {
        OpenOptions::new()
            .append(true)
            .open(self.size_output_path())
            .unwrap()
    }

    fn verify_output(&self) -> File {
        OpenOptions::new()
            .append(true)
            .open(self.verify_output_path())
            .unwrap()
    }

    fn bench(&self, k: usize) {
        type BrakedownSpec1Pcs = MultilinearBrakedown<Mersenne127, Blake2s256, BrakedownSpec1>;
        type BrakedownSpec3Pcs = MultilinearBrakedown<Mersenne127, Blake2s256, BrakedownSpec3>;
        type BrakedownSpec6Pcs = MultilinearBrakedown<Mersenne127, Blake2s256, BrakedownSpec6>;

        match self {
            System::Spec1 => {
                bench_pcs::<Mersenne127, BrakedownSpec1Pcs, Blake2sTranscript<_>>(k, *self)
            }
            System::Spec3 => {
                bench_pcs::<Mersenne127, BrakedownSpec3Pcs, Blake2sTranscript<_>>(k, *self)
            }
            System::Spec6 => {
                bench_pcs::<Mersenne127, BrakedownSpec6Pcs, Blake2sTranscript<_>>(k, *self)
            }
        }
    }
}

impl Display for System {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            System::Spec1 => write!(f, "spec1"),
            System::Spec3 => write!(f, "spec3"),
            System::Spec6 => write!(f, "spec6"),
        }
    }
}

fn parse_args() -> (Vec<System>, Range<usize>) {
    let (systems, k_range) = args().chain(Some("".to_string())).tuple_windows().fold(
        (Vec::new(), 10..25),
        |(mut systems, mut k_range), (key, value)| {
            match key.as_str() {
                "--system" => match value.as_str() {
                    "all" => systems = System::all(),
                    "spec1" => systems.push(System::Spec1),
                    "spec3" => systems.push(System::Spec3),
                    "spec6" => systems.push(System::Spec6),
                    _ => panic!("system should be one of {{all,spec1,spec3,spec6}}"),
                },
                "--k" => {
                    if let Some((start, end)) = value.split_once("..") {
                        k_range = start.parse().expect("k range start to be usize")
                            ..end.parse().expect("k range end to be usize");
                    } else {
                        k_range.start = value.parse().expect("k to be usize");
                        k_range.end = k_range.start + 1;
                    }
                }
                _ => {}
            }
            (systems, k_range)
        },
    );

    let mut systems = systems.into_iter().sorted().dedup().collect_vec();
    if systems.is_empty() {
        systems = System::all();
    }
    (systems, k_range)
}

fn create_output(systems: &[System]) {
    if !Path::new(OUTPUT_DIR).exists() {
        create_dir(OUTPUT_DIR).unwrap();
    }
    for system in systems {
        File::create(system.output_path()).unwrap();
        File::create(system.commit_output_path()).unwrap();
        File::create(system.size_output_path()).unwrap();
        File::create(system.verify_output_path()).unwrap();
    }
}

fn sample_size(k: usize) -> usize {
    if k < 16 {
        20
    } else if k < 20 {
        5
    } else {
        1
    }
}
