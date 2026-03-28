mod brakedown;
mod raa;
pub mod binary_rs;
pub use raa::{
    encode_bits, encode_bits_long, encode_bits_ser, encode_bits_ser_with_timings,
    encode_bits_with_timings, parallel_accumulator_long, Permutation, RaaEncodeTimings,
    repetition_code_long, serial_accumulator_long,
};
pub use brakedown::{
    Brakedown, BrakedownSpec, BrakedownSpec1, BrakedownSpec2, BrakedownSpec3, BrakedownSpec4,
    BrakedownSpec5, BrakedownSpec6,
};

pub trait LinearCodes<F>: Sync + Send {
    fn row_len(&self) -> usize;

    fn codeword_len(&self) -> usize;

    fn num_column_opening(&self) -> usize;

    fn num_proximity_testing(&self) -> usize;

    fn encode(&self, input: impl AsMut<[F]>);
}
