//! Applying a stored softmax rest logit to a decoder, and mapping the
//! inference library's refusal into birda's error type.

use crate::error::{Error, Result};
use birdnet_onnx::PredictionDecoder;
use std::path::Path;

/// Map an error from building a classifier or decoder for the model at `path`.
///
/// A rejected softmax rest logit becomes [`Error::ScoreCalibration`], so the
/// message names the value and the file. Anything else is a plain
/// [`Error::ClassifierBuild`].
pub fn build_error(path: &Path, error: birdnet_onnx::Error) -> Error {
    match error {
        birdnet_onnx::Error::InvalidSoftmaxRestLogit { rest_logit, reason } => {
            Error::ScoreCalibration {
                path: path.to_path_buf(),
                rest_logit,
                reason,
            }
        }
        other => Error::ClassifierBuild {
            reason: other.to_string(),
        },
    }
}

/// Apply `rest_logit` to `decoder` when there is one.
pub fn calibrated_decoder(
    decoder: PredictionDecoder,
    rest_logit: Option<f32>,
    path: &Path,
) -> Result<PredictionDecoder> {
    match rest_logit {
        Some(rest_logit) => decoder
            .with_softmax_rest_logit(rest_logit)
            .map_err(|e| build_error(path, e)),
        None => Ok(decoder),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use birdnet_onnx::{ModelConfig, ModelType};
    use std::path::PathBuf;

    fn decoder(model_type: ModelType) -> PredictionDecoder {
        let config = ModelConfig {
            model_type,
            sample_rate: 32_000,
            segment_duration: 5.0,
            sample_count: 160_000,
            num_species: 3,
            embedding_dim: None,
            predictions_index: 0,
            embeddings_index: None,
        };
        let labels = ["a", "b", "c"].map(String::from).to_vec();
        PredictionDecoder::new(config, labels, 3, None).unwrap()
    }

    /// Confidence per label for one row of logits `[2.0, 1.0, 0.0]`.
    fn confidences(decoder: &PredictionDecoder) -> Vec<f32> {
        let result = decoder.decode_batch(1, &[2.0, 1.0, 0.0], None).unwrap();
        let mut by_label = vec![0.0; 3];
        for prediction in &result[0].predictions {
            by_label[prediction.index] = prediction.confidence;
        }
        by_label
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!(
                (a - e).abs() < 1e-6,
                "got {actual:?}, expected {expected:?}"
            );
        }
    }

    #[test]
    fn test_calibrated_decoder_adds_the_rest_class_to_the_softmax() {
        let path = PathBuf::from("perch_v2_nordic.onnx");

        let with_rest = calibrated_decoder(decoder(ModelType::PerchV2), Some(3.0), &path).unwrap();
        let total: f32 = [2.0_f32, 1.0, 0.0, 3.0].iter().map(|z| z.exp()).sum();
        assert_close(
            &confidences(&with_rest),
            &[2.0_f32, 1.0, 0.0].map(|z| z.exp() / total),
        );

        let plain = calibrated_decoder(decoder(ModelType::PerchV2), None, &path).unwrap();
        let total: f32 = [2.0_f32, 1.0, 0.0].iter().map(|z| z.exp()).sum();
        assert_close(
            &confidences(&plain),
            &[2.0_f32, 1.0, 0.0].map(|z| z.exp() / total),
        );
    }

    #[test]
    fn test_a_rest_logit_on_a_sigmoid_model_is_a_score_calibration_error() {
        let path = PathBuf::from("/models/birdnet.onnx");
        let err = calibrated_decoder(decoder(ModelType::BirdNetV24), Some(1.0), &path).unwrap_err();
        assert_eq!(
            err.to_string(),
            "cannot apply softmax_rest_logit 1 to /models/birdnet.onnx: \
             model type BirdNetV24 does not use softmax"
        );
    }
}
