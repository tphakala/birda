//! Audio processing pipeline.

mod decode;
mod resample;

pub use decode::{decode_audio_file, DecodedAudio};
pub use resample::resample;
