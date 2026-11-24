//! Finding the `OpenVINO` toolkit among a user's Microsoft Store packages.
//!
//! A Store package lives in a folder named after the package full name,
//! `<Name>_<Version>_<Architecture>_<ResourceId>_<PublisherId>`, under
//! `<Program Files>\WindowsApps`. A standard user may not be able to list that
//! directory, so the package folders are also read from the registry, through
//! `reg.exe`. Everything here is pure except [`query_package_repository`].

use crate::constants::openvino as ov;
use crate::utils::text::strip_prefix_ignore_ascii_case;
use std::path::{Path, PathBuf};

/// Runs the registry query and returns its standard output, `None` when the
/// query could not run or matched nothing.
pub type RegistryQuery<'a> = &'a dyn Fn() -> Option<String>;

/// Parse a package full name into its version when it names an x64 `OpenVINO`
/// toolkit package.
///
/// The resource id is not checked: a resource package has no library in it and
/// is skipped by the caller's existence check. Component values above `u16`
/// are refused, as a package version holds four 16-bit parts.
#[must_use]
pub fn openvino_package_version(full_name: &str) -> Option<[u16; ov::MSIX_VERSION_PARTS]> {
    let fields: Vec<&str> = full_name.split('_').collect();
    if fields.len() != ov::MSIX_FIELD_COUNT {
        return None;
    }
    strip_prefix_ignore_ascii_case(fields[0], ov::MSIX_NAME_PREFIX)?;
    if !fields[2].eq_ignore_ascii_case(ov::MSIX_ARCHITECTURE) {
        return None;
    }
    let parts: Vec<u16> = fields[1]
        .split('.')
        .map(|part| part.parse().ok())
        .collect::<Option<_>>()?;
    parts.try_into().ok()
}

/// Order candidates newest version first, then by full name descending, and
/// return their folders. Independent of the order they were found in.
fn newest_first(
    mut candidates: Vec<([u16; ov::MSIX_VERSION_PARTS], String, PathBuf)>,
) -> Vec<PathBuf> {
    candidates.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
    candidates.into_iter().map(|(_, _, path)| path).collect()
}

/// The `OpenVINO` package folders among the entries of the `WindowsApps`
/// directory `dir`, newest first.
#[must_use]
pub fn from_listing(dir: &Path, names: &[String]) -> Vec<PathBuf> {
    newest_first(
        names
            .iter()
            .filter_map(|name| {
                openvino_package_version(name)
                    .map(|version| (version, name.clone(), dir.join(name)))
            })
            .collect(),
    )
}

/// Split off the first whitespace-delimited token.
fn token(text: &str) -> Option<(&str, &str)> {
    let (first, rest) = text.split_once(char::is_whitespace)?;
    Some((first, rest.trim_start()))
}

/// The package name when `key` is a direct subkey of the package repository.
fn package_of_key(key: &str) -> Option<&str> {
    let rest = strip_prefix_ignore_ascii_case(key.trim_end(), ov::PACKAGE_REPOSITORY_KEY)?
        .strip_prefix('\\')?;
    (!rest.is_empty() && !rest.contains('\\')).then_some(rest)
}

/// The `OpenVINO` package folders in the output of
/// `reg query <PACKAGE_REPOSITORY_KEY> /s /v Path`, newest first.
///
/// A key line starts a package; the indented `Path REG_SZ <folder>` line under
/// it gives the folder, which may hold spaces. Deeper subkeys, other value
/// types and the localised footer are ignored.
#[must_use]
pub fn from_registry(output: &str) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    let mut current: Option<([u16; ov::MSIX_VERSION_PARTS], String)> = None;
    for line in output.lines() {
        if line.starts_with("HKEY_") {
            current = package_of_key(line)
                .and_then(|name| openvino_package_version(name).map(|v| (v, name.to_string())));
        } else if line.starts_with(char::is_whitespace)
            && let Some((version, name)) = &current
            && let Some((value, rest)) = token(line.trim_start())
            && value.eq_ignore_ascii_case(ov::PACKAGE_PATH_VALUE)
            && let Some((kind, folder)) = token(rest)
            && kind == ov::REGISTRY_STRING_TYPE
            && !folder.trim().is_empty()
        {
            candidates.push((*version, name.clone(), PathBuf::from(folder.trim())));
        }
    }
    newest_first(candidates)
}

/// Run `reg.exe` to list every package's folder. `None` when it cannot start,
/// exits non-zero (as it does when nothing matches), or is not there.
#[must_use]
pub fn query_package_repository() -> Option<String> {
    let system_root = std::env::var_os(ov::SYSTEM_ROOT_ENV)
        .filter(|value| !value.is_empty())
        .map_or_else(|| PathBuf::from(ov::DEFAULT_SYSTEM_ROOT), PathBuf::from);
    let output = std::process::Command::new(system_root.join(ov::SYSTEM32_DIR).join(ov::REG_EXE))
        .args([
            "query",
            ov::PACKAGE_REPOSITORY_KEY,
            "/s",
            "/v",
            ov::PACKAGE_PATH_VALUE,
        ])
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_names_give_a_version_only_for_x64_openvino_packages() {
        let cases: [(&str, Option<[u16; 4]>); 8] = [
            (
                "Intel.OpenVINOToolkit.X_2026.3.0.0_x64__8wekyb3d8bbwe",
                Some([2026, 3, 0, 0]),
            ),
            (
                "intel.openvinotoolkit.x_2026.3.1.2_X64__8wekyb3d8bbwe",
                Some([2026, 3, 1, 2]),
            ),
            (
                "Intel.OpenVINOToolkit.X_2026.3.0.0_arm64__8wekyb3d8bbwe",
                None,
            ),
            ("Microsoft.Photos_2026.3.0.0_x64__8wekyb3d8bbwe", None),
            ("Intel.OpenVINOToolkit.X_2026.3.0_x64__8wekyb3d8bbwe", None),
            (
                "Intel.OpenVINOToolkit.X_2026.3.0.70000_x64__8wekyb3d8bbwe",
                None,
            ),
            ("Intel.OpenVINOToolkit.X_2026.3.0.0_x64_8wekyb3d8bbwe", None),
            ("Deleted", None),
        ];
        for (name, expected) in cases {
            assert_eq!(openvino_package_version(name), expected, "{name}");
        }
    }

    #[test]
    fn a_listing_gives_the_newest_openvino_package_first() {
        let names: Vec<String> = [
            "Intel.OpenVINOToolkit.X_2026.9.0.0_x64__8wekyb3d8bbwe",
            "Microsoft.Photos_2030.1.0.0_x64__8wekyb3d8bbwe",
            "Intel.OpenVINOToolkit.X_2026.10.0.0_x64__8wekyb3d8bbwe",
            "Intel.OpenVINOToolkit.X_2026.10.0.0_x64__zzzzzzzzzzzzz",
        ]
        .map(String::from)
        .into();
        let dir = Path::new("C:\\Apps");
        assert_eq!(
            from_listing(dir, &names),
            [
                dir.join("Intel.OpenVINOToolkit.X_2026.10.0.0_x64__zzzzzzzzzzzzz"),
                dir.join("Intel.OpenVINOToolkit.X_2026.10.0.0_x64__8wekyb3d8bbwe"),
                dir.join("Intel.OpenVINOToolkit.X_2026.9.0.0_x64__8wekyb3d8bbwe"),
            ]
        );
    }

    #[test]
    fn registry_output_yields_package_paths_newest_first() {
        let key = ov::PACKAGE_REPOSITORY_KEY;
        let output = [
            String::new(),
            format!("{key}\\Intel.OpenVINOToolkit.X_2026.2.0.0_x64__8wekyb3d8bbwe"),
            "    Path    REG_SZ    C:\\Program Files\\WindowsApps\\Intel.OpenVINOToolkit.X_2026.2.0.0_x64__8wekyb3d8bbwe".to_string(),
            String::new(),
            format!("{key}\\Microsoft.Photos_2026.9.0.0_x64__8wekyb3d8bbwe"),
            "    Path    REG_SZ    C:\\Program Files\\WindowsApps\\Microsoft.Photos".to_string(),
            String::new(),
            format!("{key}\\Intel.OpenVINOToolkit.X_2026.3.0.0_x64__8wekyb3d8bbwe"),
            "    Path    REG_SZ    D:\\Store Apps\\Intel.OpenVINOToolkit.X_2026.3.0.0_x64__8wekyb3d8bbwe".to_string(),
            String::new(),
            format!("{key}\\Intel.OpenVINOToolkit.X_2026.3.0.0_x64__8wekyb3d8bbwe\\SomeChild"),
            "    Path    REG_SZ    C:\\child".to_string(),
            String::new(),
            format!("{key}\\Intel.OpenVINOToolkit.X_2026.4.0.0_x64__8wekyb3d8bbwe"),
            "    Path    REG_DWORD    0x1".to_string(),
            String::new(),
            "End of search: 4 match(es) found.".to_string(),
        ]
        .join("\r\n");
        assert_eq!(
            from_registry(&output),
            [
                PathBuf::from(
                    "D:\\Store Apps\\Intel.OpenVINOToolkit.X_2026.3.0.0_x64__8wekyb3d8bbwe"
                ),
                PathBuf::from(
                    "C:\\Program Files\\WindowsApps\\Intel.OpenVINOToolkit.X_2026.2.0.0_x64__8wekyb3d8bbwe"
                ),
            ]
        );
    }
}
