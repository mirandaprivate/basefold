//! Implementation of linear codes specification described in [GLSTW21].
//! Most part are ported from https://github.com/conroi/lcpc with naming
//! modification to follow the notations in paper.
//!
//! [GLSTW21]: https://eprint.iacr.org/2021/1043.pdf

use crate::util::{
    arithmetic::{div_ceil, horner, steps, Field, PrimeField},
    code::LinearCodes,
    parallel::{num_threads, par_map_collect},
    Deserialize, Itertools, Serialize,
};
use rand::{seq::index::sample, RngCore, SeedableRng};
use rand_chacha::ChaCha12Rng;
use std::{
    cmp::{max, min},
    fmt::Debug,
    iter
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Brakedown<F> {
    row_len: usize,
    codeword_len: usize,
    num_column_opening: usize,
    num_proximity_testing: usize,
    a: Vec<SparseMatrix<F>>,
    b: Vec<SparseMatrix<F>>,
}

impl<F: PrimeField> Brakedown<F> {
    fn near_square_row_len<S: BrakedownSpec>(num_vars: usize, n_0: usize) -> usize {
        let poly_len = 1usize << num_vars;
        let min_row_len = (n_0 + 1).next_power_of_two().min(poly_len);
        let col_openings = S::num_column_opening();

        let approx_codeword_len = (((col_openings * poly_len) as f64).sqrt().ceil() as usize)
            .saturating_mul(2)
            .next_power_of_two()
            .max(min_row_len)
            .min(poly_len);
        let (approx_a, approx_b) = S::dimensions(F::NUM_BITS as usize, approx_codeword_len, n_0);
        let approx_full_codeword_len = S::codeword_len_from_dimensions(&approx_a, &approx_b);
        let degree_tests = S::num_proximity_testing_from_codeword_len(
            F::NUM_BITS as usize,
            approx_full_codeword_len,
        )
        .max(1);
        let near_square_row_len = (((col_openings * poly_len) as f64 / degree_tests as f64)
            .sqrt()
            .ceil() as usize)
            .next_power_of_two();

        near_square_row_len.max(min_row_len).min(poly_len)
    }

    pub fn proof_size<S: BrakedownSpec>(n_0: usize, c: usize, r: usize) -> usize {
        let log2_q = F::NUM_BITS as usize;
        let num_ldt = S::num_proximity_testing(log2_q, c, n_0);
        (1 + num_ldt) * c + S::num_column_opening() * r
    }

    pub fn new_multilinear<S: BrakedownSpec>(
        num_vars: usize,
        n_0: usize,
        mut rng: impl RngCore,
    ) -> Self {
        assert!(1 << num_vars > n_0);

        let log2_q = F::NUM_BITS as usize;
        let row_len = Self::near_square_row_len::<S>(num_vars, n_0);
        let (a_dimensions, b_dimensions) = S::dimensions(log2_q, row_len, n_0);
        let codeword_len = S::codeword_len_from_dimensions(&a_dimensions, &b_dimensions);

        let num_column_opening = S::num_column_opening();
        let num_proximity_testing =
            S::num_proximity_testing_from_codeword_len(log2_q, codeword_len);
        let mut matrix_seed = [0u8; 32];
        rng.fill_bytes(&mut matrix_seed);
        let (a, b) = S::matrices_from_dimensions(
            a_dimensions,
            b_dimensions,
            ChaCha12Rng::from_seed(matrix_seed),
        );

        Self {
            row_len,
            codeword_len,
            num_column_opening,
            num_proximity_testing,
            a,
            b,
        }
    }
}

impl<F: PrimeField> LinearCodes<F> for Brakedown<F> {
    fn row_len(&self) -> usize {
        self.row_len
    }

    fn codeword_len(&self) -> usize {
        self.codeword_len
    }

    fn num_column_opening(&self) -> usize {
        self.num_column_opening
    }

    fn num_proximity_testing(&self) -> usize {
        self.num_proximity_testing
    }

    fn encode(&self, mut target: impl AsMut<[F]>) {
        let target = target.as_mut();
        assert_eq!(target.len(), self.codeword_len);

        let mut input_offset = 0;
        self.a[..self.a.len() - 1].iter().for_each(|a| {
            let (input, output) = target[input_offset..].split_at_mut(a.dimension.n);
            a.dot_into(input, &mut output[..a.dimension.m]);
            input_offset += a.dimension.n;
        });

        let a_last = self.a.last().unwrap();
        let b_last = self.b.last().unwrap();
        let (input, output) = target[input_offset..].split_at_mut(a_last.dimension.n);
        let tmp = a_last.dot(input);
        reed_solomon_into(&tmp, &mut output[..b_last.dimension.n]);
        let mut output_offset = input_offset + a_last.dimension.n + b_last.dimension.n;
        input_offset += a_last.dimension.n + a_last.dimension.m;

        self.a
            .iter()
            .rev()
            .zip(self.b.iter().rev())
            .for_each(|(a, b)| {
                input_offset -= a.dimension.m;
                let (input, output) = target.split_at_mut(output_offset);
                b.dot_into(
                    &input[input_offset..input_offset + b.dimension.n],
                    &mut output[..b.dimension.m],
                );
                output_offset += b.dimension.m;
            });

        if cfg!(feature = "sanity-check") {
            assert_eq!(input_offset, self.a[0].dimension.n);
            assert_eq!(output_offset, target.len());
        }
    }
}

pub trait BrakedownSpec: Debug {
    const LAMBDA: f64;
    const ALPHA: f64;
    const BETA: f64;
    const R: f64;

    fn delta() -> f64 {
        Self::BETA / Self::R
    }

    fn mu() -> f64 {
        Self::R - 1f64 - Self::R * Self::ALPHA
    }

    fn nu() -> f64 {
        Self::BETA + Self::ALPHA * Self::BETA + 0.03
    }

    fn c_n(n: usize) -> usize {
        let alpha = Self::ALPHA;
        let beta = Self::BETA;
        let n = n as f64;
        min(
            max(ceil(1.28 * beta * n), ceil(beta * n) + 4),
            ceil(
                ((110.0 / n) + h(beta) + alpha * h(1.28 * beta / alpha))
                    / (beta * (alpha / (1.28 * beta)).log2()),
            ),
        )
    }

    fn d_n(log2_q: usize, n: usize) -> usize {
        let alpha = Self::ALPHA;
        let beta = Self::BETA;
        let r = Self::R;
        let mu = Self::mu();
        let nu = Self::nu();
        let log2_q = log2_q as f64;
        let n = n as f64;
        min(
            ceil((2.0 * beta + ((r - 1.0) + 110.0 / n) / log2_q) * n),
            ceil(
                (r * alpha * h(beta / r) + mu * h(nu / mu) + 110.0 / n)
                    / (alpha * beta * (mu / nu).log2()),
            ),
        )
    }

    fn num_column_opening() -> usize {
//	1
        let numc = ceil(-Self::LAMBDA / (1.0 - Self::delta() / 3.0).log2());
//	println!("num c {:?}", numc);
	numc
    }

    fn num_proximity_testing(log2_q: usize, n: usize, n_0: usize) -> usize {
        let (a, b) = Self::dimensions(log2_q, n, n_0);
        Self::num_proximity_testing_from_codeword_len(
            log2_q,
            Self::codeword_len_from_dimensions(&a, &b),
        )
    }

    fn num_proximity_testing_from_codeword_len(log2_q: usize, codeword_len: usize) -> usize {
        ceil(Self::LAMBDA / (log2_q as f64 - (codeword_len as f64).log2()))
    }

    fn dimensions(
        log2_q: usize,
        n: usize,
        n_0: usize,
    ) -> (Vec<SparseMatrixDimension>, Vec<SparseMatrixDimension>) {
        assert!(n > n_0);

        let a = iter::successors(Some(n), |n| Some(ceil(*n as f64 * Self::ALPHA)))
            .tuple_windows()
            .map(|(n, m)| SparseMatrixDimension::new(n, m, min(Self::c_n(n), m)))
            .take_while(|a| a.n > n_0)
            .collect_vec();
        let b_jobs = a.to_vec();
        let b: Vec<SparseMatrixDimension> = par_map_collect(b_jobs, |a| {
                let n_prime = ceil(a.m as f64 * Self::R);
                let m_prime = ceil(a.n as f64 * Self::R) - a.n - n_prime;
                SparseMatrixDimension::new(n_prime, m_prime, min(Self::d_n(log2_q, a.n), m_prime))
            });

        (a, b)
    }

    fn codeword_len(log2_q: usize, n: usize, n_0: usize) -> usize {
        let (a, b) = Self::dimensions(log2_q, n, n_0);
        Self::codeword_len_from_dimensions(&a, &b)
    }

    fn codeword_len_from_dimensions(
        a: &[SparseMatrixDimension],
        b: &[SparseMatrixDimension],
    ) -> usize {
        assert!(!a.is_empty());
        iter::empty()
            .chain(Some(a[0].n))
            .chain(a[..a.len() - 1].iter().map(|a| a.m))
            .chain(Some(b.last().unwrap().n))
            .chain(b.iter().map(|b| b.m))
            .sum()
    }

    fn matrices<F: PrimeField>(
        log2_q: usize,
        n: usize,
        n_0: usize,
        rng: impl RngCore,
    ) -> (Vec<SparseMatrix<F>>, Vec<SparseMatrix<F>>) {
        let (a, b) = Self::dimensions(log2_q, n, n_0);
        Self::matrices_from_dimensions(a, b, rng)
    }

    fn matrices_from_dimensions<F: PrimeField>(
        a: Vec<SparseMatrixDimension>,
        b: Vec<SparseMatrixDimension>,
        mut rng: impl RngCore,
    ) -> (Vec<SparseMatrix<F>>, Vec<SparseMatrix<F>>) {
        let matrix_seeds = iter::repeat_with(|| {
            let mut a_seed = [0u8; 32];
            let mut b_seed = [0u8; 32];
            rng.fill_bytes(&mut a_seed);
            rng.fill_bytes(&mut b_seed);
            (a_seed, b_seed)
        })
        .take(a.len())
        .collect::<Vec<_>>();

        let matrix_jobs = a
            .into_iter()
            .zip(b)
            .zip(matrix_seeds)
            .collect::<Vec<_>>();

        let matrices: Vec<(SparseMatrix<F>, SparseMatrix<F>)> =
            par_map_collect(matrix_jobs, |((a, b), (a_seed, b_seed))| {
            (
                SparseMatrix::new(a, ChaCha12Rng::from_seed(a_seed)),
                SparseMatrix::new(b, ChaCha12Rng::from_seed(b_seed)),
            )
        });

        matrices.into_iter().unzip()
    }
}

macro_rules! impl_spec_128 {
    ($(($name:ident, $alpha:literal, $beta:literal, $r:literal)),*) => {
        $(
            #[derive(Debug)]
            pub struct $name;
            impl BrakedownSpec for $name {
                const LAMBDA: f64 = 80.0;
                const ALPHA: f64 = $alpha;
                const BETA: f64 = $beta;
                const R: f64 = $r;
            }
        )*
    };
}

// Figure 2 in [GLSTW21](https://eprint.iacr.org/2021/1043.pdf).
impl_spec_128!(
    (BrakedownSpec1, 0.1195, 0.0284, 1.420),
    (BrakedownSpec2, 0.1380, 0.0444, 1.470),
    (BrakedownSpec3, 0.1780, 0.0610, 1.521),
    (BrakedownSpec4, 0.2000, 0.0820, 1.640),
    (BrakedownSpec5, 0.2110, 0.0970, 1.616),
    (BrakedownSpec6, 0.2500, 0.1250, 2.000)
);

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SparseMatrixDimension {
    n: usize,
    m: usize,
    d: usize,
}

impl SparseMatrixDimension {
    fn new(n: usize, m: usize, d: usize) -> Self {
        Self { n, m, d }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SparseMatrix<F> {
    dimension: SparseMatrixDimension,
    cells: Vec<(usize, F)>,
}

impl<F: PrimeField> SparseMatrix<F> {
    fn new(dimension: SparseMatrixDimension, mut rng: impl RngCore) -> Self {
        let chunk_size = div_ceil(dimension.n, num_threads()).max(1);
        let row_jobs = (0..dimension.n)
            .step_by(chunk_size)
            .map(|start| {
                let mut seed = [0u8; 32];
                rng.fill_bytes(&mut seed);
                let row_count = (dimension.n - start).min(chunk_size);
                (seed, row_count)
            })
            .collect::<Vec<_>>();
        let row_chunks: Vec<Vec<(usize, F)>> = par_map_collect(row_jobs, |(seed, row_count)| {
            let mut row_rng = ChaCha12Rng::from_seed(seed);
            (0..row_count)
                .flat_map(|_| {
                    sample(&mut row_rng, dimension.m, dimension.d)
                        .into_iter()
                        .map(|column| (column, fast_random_coeff(&mut row_rng)))
                        .collect_vec()
                })
                .collect()
        });
        let cells = row_chunks.into_iter().flatten().collect();
        Self { dimension, cells }
    }
}

impl<F: Field> SparseMatrix<F> {
    fn rows(&self) -> impl Iterator<Item = &[(usize, F)]> {
        self.cells.chunks(self.dimension.d)
    }

    fn dot_into(&self, array: &[F], mut target: impl AsMut<[F]>) {
        let target = target.as_mut();
        assert_eq!(self.dimension.n, array.len());
        assert_eq!(self.dimension.m, target.len());

        self.rows().zip(array.iter()).for_each(|(cells, item)| {
            cells.iter().for_each(|(column, coeff)| {
                target[*column] += *item * coeff;
            })
        });
    }

    fn dot(&self, array: &[F]) -> Vec<F> {
        let mut target = vec![F::ZERO; self.dimension.m];
        self.dot_into(array, &mut target);
        target
    }
}

fn reed_solomon_into<F: Field>(input: &[F], mut target: impl AsMut<[F]>) {
    target
        .as_mut()
        .iter_mut()
        .zip(steps(F::ONE))
        .for_each(|(target, x)| *target = horner(input, &x));
}

fn fast_random_coeff<F: PrimeField>(mut rng: impl RngCore) -> F {
    let byte_len = (F::NUM_BITS as usize).next_power_of_two() / 8;
    assert!(byte_len <= 64);
    let mut bytes = [0u8; 64];
    rng.fill_bytes(&mut bytes[..byte_len]);

    let radix = F::from(256);
    bytes[..byte_len]
        .iter()
        .fold(F::ZERO, |acc, byte| acc * radix + F::from(u64::from(*byte)))
}

// H(p) = -p \log_2(p) - (1 - p) \log_2(1 - p)
fn h(p: f64) -> f64 {
    assert!(0f64 < p && p < 1f64);
    let one_minus_p = 1f64 - p;
    -p * p.log2() - one_minus_p * one_minus_p.log2()
}

fn ceil(v: f64) -> usize {
    v.ceil() as usize
}

#[cfg(test)]
mod test {
    use crate::util::code::{
        BrakedownSpec, BrakedownSpec1, BrakedownSpec2, BrakedownSpec3, BrakedownSpec4,
        BrakedownSpec5, BrakedownSpec6,
    };

    fn assert_spec_correct<S: BrakedownSpec>(
        log2_q: usize,
        delta: f64,
        c_n: usize,
        d_n: usize,
        num_column_opening: usize,
        num_proximity_testing: usize,
    ) {
        let n = 1 << 30;
        let n_0 = 30;

	
        assert!(S::delta() - delta < 1e-3);
        assert_eq!(S::c_n(n), c_n);
        assert_eq!(S::d_n(log2_q, n), d_n);
        assert_eq!(S::num_column_opening(), num_column_opening);
        assert_eq!(
            S::num_proximity_testing(log2_q, n, n_0),
            num_proximity_testing
        );
    }

    #[rustfmt::skip]
    #[test]
    fn spec_127_bit_field() {
        assert_spec_correct::<BrakedownSpec1>(127, 0.02,  6, 33, 8291, 1);
        assert_spec_correct::<BrakedownSpec2>(127, 0.03,  7, 26, 5480, 1);
        assert_spec_correct::<BrakedownSpec3>(127, 0.04,  7, 22, 4121, 1);
        assert_spec_correct::<BrakedownSpec4>(127, 0.05,  8, 19, 3300, 1);
        assert_spec_correct::<BrakedownSpec5>(127, 0.06,  9, 21, 2744, 1);
        assert_spec_correct::<BrakedownSpec6>(127, 0.0625, 10, 15, 2634, 1);
    }

    #[rustfmt::skip]
    #[test]
    fn spec_254_bit_field() {
        assert_spec_correct::<BrakedownSpec1>(254, 0.02,  6, 33, 8291, 1);
        assert_spec_correct::<BrakedownSpec2>(254, 0.03,  7, 26, 5480, 1);
        assert_spec_correct::<BrakedownSpec3>(254, 0.04,  7, 22, 4121, 1);
        assert_spec_correct::<BrakedownSpec4>(254, 0.05,  8, 19, 3300, 1);
        assert_spec_correct::<BrakedownSpec5>(254, 0.06,  9, 21, 2744, 1);
        assert_spec_correct::<BrakedownSpec6>(254, 0.0625, 10, 15, 2634, 1);
    }
}
