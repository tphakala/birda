//! License display and acceptance prompts.

#![allow(clippy::print_stdout)]

use super::types::LicenseInfo;
use crate::error::Result;
use std::io::{self, Write};

/// Identity of a downloadable asset, for the licence prompt.
///
/// Covers both classifier models and shared assets such as the `BirdNET`
/// Geomodel, which is not a [`super::types::ModelEntry`].
#[derive(Debug, Clone, Copy)]
pub struct LicensedAsset<'a> {
    /// Display name, e.g. "`BirdNET` Geomodel v3.0.2".
    pub name: &'a str,
    /// Organization/author.
    pub vendor: &'a str,
    /// Version string.
    pub version: &'a str,
    /// License terms.
    pub license: &'a LicenseInfo,
}

/// Display license information and prompt for acceptance.
///
/// When `interactive` is `false`, skips the user prompt and returns `Ok(true)`
/// (auto-accept). This is used for non-TTY or structured output modes.
///
/// Returns `Ok(true)` if user accepts, `Ok(false)` if user declines.
pub fn prompt_license_acceptance(asset: LicensedAsset<'_>, interactive: bool) -> Result<bool> {
    if !interactive {
        return Ok(true);
    }

    println!("Model: {}", asset.name);
    println!("Vendor: {}", asset.vendor);
    println!("Version: {}", asset.version);
    println!();

    display_license_summary(asset.license, asset.vendor);

    println!();
    print!("Type 'accept' to continue, or anything else to cancel: ");
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    Ok(input.trim().eq_ignore_ascii_case("accept"))
}

/// Display license summary with key restrictions.
fn display_license_summary(license: &LicenseInfo, vendor: &str) {
    println!("License: {}", license.r#type);
    println!(
        "  Commercial use: {}",
        if license.commercial_use {
            "Allowed"
        } else {
            "Not allowed"
        }
    );
    println!(
        "  Attribution required: {}",
        if license.attribution_required {
            "Yes"
        } else {
            "No"
        }
    );
    println!(
        "  Share-alike required: {}",
        if license.share_alike { "Yes" } else { "No" }
    );
    println!();
    println!("Full license text:");
    println!("{}", license.url);
    println!();

    // Display key obligations
    if !license.commercial_use || license.attribution_required || license.share_alike {
        println!("By using this model, you agree to:");

        if !license.commercial_use {
            println!("  • Use for non-commercial purposes only");
        }

        if license.attribution_required {
            println!("  • Provide attribution to {vendor}");
        }

        if license.share_alike {
            println!(
                "  • Share derivatives under the same license ({})",
                license.r#type
            );
        }

        println!();
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // Test setup code - panics are acceptable
mod tests {
    use super::*;

    fn geomodel_license() -> LicenseInfo {
        LicenseInfo {
            r#type: "CC-BY-SA-4.0".into(),
            url: "https://creativecommons.org/licenses/by-sa/4.0/".into(),
            commercial_use: true,
            attribution_required: true,
            share_alike: true,
        }
    }

    #[test]
    fn test_prompt_auto_accepts_when_not_interactive() {
        let license = geomodel_license();
        let asset = LicensedAsset {
            name: "BirdNET Geomodel v3.0.2",
            vendor: "Cornell Lab of Ornithology",
            version: "3.0.2",
            license: &license,
        };

        assert!(prompt_license_acceptance(asset, false).unwrap());
    }

    #[test]
    fn test_display_license_summary_share_alike() {
        // The geomodel is CC BY-SA, unlike the CC BY-NC-SA classifier models,
        // so the share-alike obligation must render.
        display_license_summary(&geomodel_license(), "Cornell Lab of Ornithology");
    }

    #[test]
    fn test_display_license_summary_noncommercial() {
        let license = LicenseInfo {
            r#type: "CC-BY-NC-SA-4.0".into(),
            url: "https://creativecommons.org/licenses/by-nc-sa/4.0/".into(),
            commercial_use: false,
            attribution_required: true,
            share_alike: true,
        };

        // This test verifies the function exists and can be called
        // We can't easily capture stdout in unit tests, so we just verify compilation
        display_license_summary(&license, "Test Vendor");
    }

    #[test]
    fn test_display_license_summary_commercial() {
        let license = LicenseInfo {
            r#type: "Apache-2.0".into(),
            url: "https://www.apache.org/licenses/LICENSE-2.0".into(),
            commercial_use: true,
            attribution_required: true,
            share_alike: false,
        };

        // Verify function can be called
        display_license_summary(&license, "Test Vendor");
    }

    #[test]
    fn test_display_license_summary_permissive() {
        let license = LicenseInfo {
            r#type: "MIT".into(),
            url: "https://opensource.org/licenses/MIT".into(),
            commercial_use: true,
            attribution_required: false,
            share_alike: false,
        };

        // Verify function can be called
        display_license_summary(&license, "Test Vendor");
    }
}
