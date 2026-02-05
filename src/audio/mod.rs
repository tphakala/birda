//! Audio processing pipeline.

mod chunker;
mod decode;
mod resample;

pub use chunker::{AudioChunk, chunk_audio};
pub use decode::{
    DecodedAudio, RawSegment, StreamingDecoder, decode_audio_file, get_audio_duration,
};
pub use resample::{resample, resample_chunk};
