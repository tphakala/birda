//! Score calibration of regional model installs.

use super::{ModelConfig, ModelType};

/// The warning for a regional Perch install that has no stored calibration,
/// with the command that installs it again.
///
/// `None` for every other entry: a global install, a model added by hand, a
/// `BirdNET` regional install, and a Perch entry with a stored value. An entry
/// with a region but no registry id was written by hand, since `models install`
/// writes both together, and there is no install command to give for it.
#[must_use]
pub fn uncalibrated_region_warning(model: &ModelConfig, model_name: &str) -> Option<String> {
    if model.model_type != ModelType::PerchV2 || model.softmax_rest_logit.is_some() {
        return None;
    }
    let region = model.region.as_deref()?;
    let id = model.registry_id.as_deref()?;
    Some(format!(
        "model '{model_name}' has no stored score calibration, so its scores are \
         uncalibrated and run higher than the global Perch model's. Reinstall it to \
         fix this: birda models install {id} --region {region}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn uncalibrated_nordic() -> ModelConfig {
        ModelConfig {
            path: PathBuf::from("perch_v2_nordic.onnx"),
            labels: PathBuf::from("perch_v2_nordic-labels.txt"),
            model_type: ModelType::PerchV2,
            meta_model: None,
            bsg_calibration: None,
            bsg_migration: None,
            bsg_distribution_maps: None,
            registry_id: Some("perch-v2".to_string()),
            installed_version: Some("2".to_string()),
            installed_build: Some(1),
            region: Some("nordic".to_string()),
            variant: Some("no-dft-fp32".to_string()),
            softmax_rest_logit: None,
            accepted_documents: Vec::new(),
        }
    }

    #[test]
    fn test_uncalibrated_regional_perch_install_names_the_reinstall_command() {
        assert_eq!(
            uncalibrated_region_warning(&uncalibrated_nordic(), "perch-v2-nordic").as_deref(),
            Some(
                "model 'perch-v2-nordic' has no stored score calibration, so its scores are \
                 uncalibrated and run higher than the global Perch model's. Reinstall it to \
                 fix this: birda models install perch-v2 --region nordic"
            )
        );
    }

    #[test]
    fn test_each_missing_condition_silences_the_warning() {
        type Edit = fn(&mut ModelConfig);
        let rows: [(&str, Edit); 4] = [
            ("a stored value", |m| m.softmax_rest_logit = Some(9.75)),
            ("no region (global or hand-made)", |m| m.region = None),
            ("no registry id (hand-made)", |m| m.registry_id = None),
            ("a BirdNET model type", |m| {
                m.model_type = ModelType::BirdnetV30;
            }),
        ];
        for (name, edit) in rows {
            let mut model = uncalibrated_nordic();
            edit(&mut model);
            assert_eq!(
                uncalibrated_region_warning(&model, "perch-v2-nordic"),
                None,
                "{name} must silence the warning"
            );
        }
    }
}
