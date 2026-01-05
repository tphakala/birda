//! Audio processing pipeline.

mod chunker;
mod decode;
mod resample;

pub use chunker::{AudioChunk, chunk_audio};
pub use decode::{DecodedAudio, RawSegment, StreamingDecoder, decode_audio_file};
pub use resample::{resample, resample_chunk};
