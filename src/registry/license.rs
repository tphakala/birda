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
/// When `assume_yes` is set the terms are still DISPLAYED and only the question
/// is skipped. Suppressing the display as well would mean a user at a terminal
/// accepts a licence they were never shown, which is worse than the non-TTY
/// case where no human is present to read it. The classifiers are CC BY-NC-SA
/// (non-commercial) and the geomodel is CC BY-SA (share-alike), so what is
/// being agreed to differs by asset and is worth printing.
///
/// Returns `Ok(true)` if user accepts, `Ok(false)` if user declines.
pub fn prompt_license_acceptance(
    asset: LicensedAsset<'_>,
    interactive: bool,
    assume_yes: bool,
) -> Result<bool> {
    if !interactive {
        return Ok(true);
    }

    println!("Model: {}", asset.name);
    println!("Vendor: {}", asset.vendor);
    println!("Version: {}", asset.version);
    println!();

    display_license_summary(asset.license, asset.vendor);

    if assume_yes {
        println!("Accepted via --yes.");
        return Ok(true);
    }

    println!();
    print!("Type 'accept' to continue, or anything else to cancel: ");
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    Ok(input.trim().eq_ignore_ascii_case("accept"))
}

/// Display license summary with key restrictions.
fn display_license_summary(license: &LicenseInfo, vendor: &str) {
    print!("{}", license_summary(license, vendor));
}

/// Render the license summary.
///
/// Split from the printing so it can be asserted on. The tests for this
/// previously called the printing version and asserted nothing, which meant a
/// summary that silently dropped the share-alike obligation would still have
/// passed a green suite (#291).
fn license_summary(license: &LicenseInfo, vendor: &str) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    let yes_no = |flag: bool| if flag { "Yes" } else { "No" };

    let _ = writeln!(out, "License: {}", license.r#type);
    let _ = writeln!(
        out,
        "  Commercial use: {}",
        if license.commercial_use {
            "Allowed"
        } else {
            "Not allowed"
        }
    );
    let _ = writeln!(
        out,
        "  Attribution required: {}",
        yes_no(license.attribution_required)
    );
    let _ = writeln!(
        out,
        "  Share-alike required: {}",
        yes_no(license.share_alike)
    );
    let _ = writeln!(out);
    let _ = writeln!(out, "Full license text:");
    let _ = writeln!(out, "{}", license.url);
    let _ = writeln!(out);

    // Display key obligations
    if !license.commercial_use || license.attribution_required || license.share_alike {
        let _ = writeln!(out, "By using this model, you agree to:");

        if !license.commercial_use {
            let _ = writeln!(out, "  • Use for non-commercial purposes only");
        }

        if license.attribution_required {
            let _ = writeln!(out, "  • Provide attribution to {vendor}");
        }

        if license.share_alike {
            let _ = writeln!(
                out,
                "  • Share derivatives under the same license ({})",
                license.r#type
            );
        }

        let _ = writeln!(out);
    }

    out
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

        assert!(prompt_license_acceptance(asset, false, false).unwrap());
    }

    #[test]
    fn test_license_summary_states_the_share_alike_obligation() {
        // The geomodel is CC BY-SA, unlike the CC BY-NC-SA classifier models,
        // so the share-alike obligation must render. This previously called the
        // printing function and asserted nothing, so a summary that dropped the
        // obligation entirely would still have passed.
        let summary = license_summary(&geomodel_license(), "Cornell Lab of Ornithology");

        assert!(
            summary.contains("Share-alike required: Yes"),
            "must state the obligation, got:\n{summary}"
        );
        assert!(
            summary.contains("Share derivatives under the same license (CC-BY-SA-4.0)"),
            "must name the licence in the obligation list, got:\n{summary}"
        );
        assert!(
            summary.contains("Commercial use: Allowed"),
            "CC BY-SA permits commercial use, unlike the classifiers, and \
             conflating the two would misinform a commercial user, got:\n{summary}"
        );
    }

    #[test]
    fn test_license_summary_states_the_non_commercial_restriction() {
        let license = LicenseInfo {
            r#type: "CC-BY-NC-SA-4.0".into(),
            url: "https://creativecommons.org/licenses/by-nc-sa/4.0/".into(),
            commercial_use: false,
            attribution_required: true,
            share_alike: true,
        };

        let summary = license_summary(&license, "Test Vendor");

        assert!(
            summary.contains("Commercial use: Not allowed"),
            "got:\n{summary}"
        );
        assert!(
            summary.contains("Use for non-commercial purposes only"),
            "the restriction must appear in the obligation list, got:\n{summary}"
        );
        assert!(
            summary.contains("Provide attribution to Test Vendor"),
            "got:\n{summary}"
        );
    }

    #[test]
    fn test_license_summary_reports_a_permissive_licence_without_obligations() {
        let license = LicenseInfo {
            r#type: "MIT".into(),
            url: "https://opensource.org/licenses/MIT".into(),
            commercial_use: true,
            attribution_required: false,
            share_alike: false,
        };

        let summary = license_summary(&license, "Test Vendor");

        assert!(
            summary.contains("Commercial use: Allowed"),
            "got:\n{summary}"
        );
        assert!(
            !summary.contains("By using this model, you agree to:"),
            "a licence with no obligations must not print an empty obligation \
             list, got:\n{summary}"
        );
    }

    #[test]
    fn test_license_summary_always_includes_the_full_text_url() {
        // The summary is a summary; the URL is the only route to the actual
        // terms, so it must survive regardless of which flags are set.
        let summary = license_summary(&geomodel_license(), "Cornell Lab of Ornithology");

        assert!(
            summary.contains("https://creativecommons.org/licenses/by-sa/4.0/"),
            "got:\n{summary}"
        );
    }
}
