//! Audio processing pipeline.

mod chunker;
mod decode;
mod resample;

pub use chunker::{chunk_audio, AudioChunk};
pub use decode::{decode_audio_file, DecodedAudio};
pub use resample::resample;
