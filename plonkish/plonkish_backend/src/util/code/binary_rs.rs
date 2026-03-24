use crate::util::arithmetic::PrimeField;
use core::marker::PhantomData;
use std::ops::Index;

#[derive(Clone, Debug)]
pub struct BinarySubspace<F> {
    dim: usize,
    _marker: PhantomData<F>,
}

impl<F> BinarySubspace<F> {
    pub fn with_dim(dim: usize) -> Result<Self, String> {
        if dim == 0 {
            return Err("binary subspace dimension must be positive".to_string());
        }
        Ok(Self {
            dim,
            _marker: PhantomData,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}

pub trait TwiddleAccess<F> {
    fn get_odd_from_even(&self, even: F) -> F;
    fn get_pair(&self, level: usize, index: usize) -> (F, F);
}

#[derive(Clone, Debug)]
pub struct OnTheFlyTwiddleAccess<F> {
    level_shift: F,
}

impl<F: PrimeField> OnTheFlyTwiddleAccess<F> {
    fn shift_for_level(level: usize) -> F {
        F::from((level + 1) as u64)
    }

    pub fn generate(subspace: &BinarySubspace<F>) -> Result<TwiddleAccessTable<F>, String> {
        let accesses = (0..subspace.dim().saturating_sub(1))
            .map(|level| Self {
                level_shift: Self::shift_for_level(level),
            })
            .collect();
        Ok(TwiddleAccessTable { accesses })
    }
}

impl<F: PrimeField> TwiddleAccess<F> for OnTheFlyTwiddleAccess<F> {
    fn get_odd_from_even(&self, even: F) -> F {
        even + self.level_shift
    }

    fn get_pair(&self, _level: usize, index: usize) -> (F, F) {
        let even = F::from(index as u64);
        (even, self.get_odd_from_even(even))
    }
}

#[derive(Clone, Debug)]
pub struct TwiddleAccessTable<F> {
    accesses: Vec<OnTheFlyTwiddleAccess<F>>,
}

impl<F> Index<usize> for TwiddleAccessTable<F> {
    type Output = OnTheFlyTwiddleAccess<F>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.accesses[index]
    }
}
