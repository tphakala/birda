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
/// `interactive` is checked first and dominates. When it is `false` this
/// returns `Ok(true)` immediately, printing nothing at all: that is the non-TTY
/// and structured-output path, where no human is present to read the terms and
/// blocking on a prompt would hang a pipeline.
///
/// `assume_yes` therefore only has an effect when `interactive` is `true`, and
/// there it skips the QUESTION while still displaying the terms. Suppressing
/// the display as well would mean a user at a terminal accepts a licence they
/// were never shown, which is worse than the non-TTY case above because someone
/// is there to read it. The classifiers are CC BY-NC-SA (non-commercial) and
/// the geomodel is CC BY-SA (share-alike), so what is being agreed to differs
/// by asset and is worth printing.
///
/// In short: `!interactive` shows nothing and accepts; `interactive &&
/// assume_yes` shows the terms and accepts; `interactive && !assume_yes` shows
/// the terms and asks.
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

impl From<&LicenseInfo> for crate::output::LicenseDetails {
    fn from(license: &LicenseInfo) -> Self {
        Self {
            r#type: license.r#type.clone(),
            url: license.url.clone(),
            commercial_use: license.commercial_use,
            attribution_required: license.attribution_required,
            share_alike: license.share_alike,
        }
    }
}

/// "Yes" or "No", the one vocabulary every licence field is rendered in.
const fn yes_no(flag: bool) -> &'static str {
    if flag { "Yes" } else { "No" }
}

/// Render the `License:` block of a `models info` view.
///
/// One renderer for classifiers, the range filter and the bat catalog. The
/// three inline copies it replaced had already drifted: the bat view left out
/// the share-alike line the other two printed. Ends with a blank line.
#[must_use]
pub fn license_details(license: &LicenseInfo) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    let _ = writeln!(out, "License:");
    let _ = writeln!(out, "  Type: {}", license.r#type);
    let _ = writeln!(out, "  URL: {}", license.url);
    let _ = writeln!(out, "  Commercial use: {}", yes_no(license.commercial_use));
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
    out
}

/// Render a licence identifier with the restrictions that apply to it.
///
/// One renderer for classifiers and the range filter alike. Listing them
/// separately taught a falsehood: the classifier loop showed only
/// `(non-commercial)` and the range filter showed only `(share-alike)`, so
/// `birdnet-v24` and `bsg-fi-v44` listed without a share-alike note even though
/// both carry that obligation. Whichever restrictions apply are now named on
/// every entry.
pub fn license_line(license: &LicenseInfo) -> String {
    let mut notes = Vec::new();
    if !license.commercial_use {
        notes.push("non-commercial");
    }
    if license.share_alike {
        notes.push("share-alike");
    }

    if notes.is_empty() {
        license.r#type.clone()
    } else {
        format!("{} ({})", license.r#type, notes.join(", "))
    }
}

/// One-paragraph notice for an asset installed alongside something the user
/// asked for by name.
///
/// A classifier install shows the classifier's licence and then fetches the
/// geomodel as a side effect. Its terms differ (CC BY-SA permits commercial use
/// and binds share-alike, where the classifiers are non-commercial), so a user
/// who accepted only the classifier's terms would otherwise never see them. This
/// discloses without a second prompt: the typed `accept` gate stays on
/// `birda models install geomodel`, the command that installs it deliberately.
#[must_use]
pub fn side_install_notice(asset: LicensedAsset<'_>) -> String {
    format!(
        "Also installing {} ({}), which is licensed separately: {}.\nTerms: {}\n",
        asset.name,
        asset.vendor,
        license_line(asset.license),
        asset.license.url
    )
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

    let _ = writeln!(out, "License: {}", license.r#type);
    let _ = writeln!(out, "  Commercial use: {}", yes_no(license.commercial_use));
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
            summary.contains("Commercial use: Yes"),
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

        assert!(summary.contains("Commercial use: No"), "got:\n{summary}");
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

        assert!(summary.contains("Commercial use: Yes"), "got:\n{summary}");
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

    fn license(commercial_use: bool, share_alike: bool) -> LicenseInfo {
        LicenseInfo {
            r#type: "TEST-1.0".into(),
            url: "https://example.com/licence".into(),
            commercial_use,
            attribution_required: true,
            share_alike,
        }
    }

    #[test]
    fn test_license_line_names_every_restriction_that_applies() {
        // The defect this replaced: the classifier loop showed only
        // "(non-commercial)" and the range filter only "(share-alike)", so
        // birdnet-v24 and bsg-fi-v44 listed with no share-alike note despite
        // carrying that obligation. Both restrictions must show together.
        let line = license_line(&license(false, true));

        assert!(line.contains("non-commercial"), "got: {line}");
        assert!(line.contains("share-alike"), "got: {line}");
    }

    #[test]
    fn test_license_line_names_share_alike_on_a_commercial_licence() {
        // The geomodel's shape: CC BY-SA permits commercial use but still binds
        // share-alike, so the note must not be suppressed by commercial_use.
        let line = license_line(&license(true, true));

        assert!(!line.contains("non-commercial"), "got: {line}");
        assert!(line.contains("share-alike"), "got: {line}");
    }

    #[test]
    fn test_license_line_adds_nothing_for_an_unrestricted_licence() {
        assert_eq!(license_line(&license(true, false)), "TEST-1.0");
    }

    #[test]
    fn test_license_details_renders_every_field_in_one_vocabulary() {
        assert_eq!(
            license_details(&geomodel_license()),
            "License:\n  Type: CC-BY-SA-4.0\n  URL: https://creativecommons.org/licenses/by-sa/4.0/\n  \
             Commercial use: Yes\n  Attribution required: Yes\n  Share-alike required: Yes\n\n"
        );
        assert_eq!(
            license_details(&license(false, false)),
            "License:\n  Type: TEST-1.0\n  URL: https://example.com/licence\n  \
             Commercial use: No\n  Attribution required: Yes\n  Share-alike required: No\n\n"
        );
    }

    #[test]
    fn test_side_install_notice_names_the_asset_and_its_own_terms() {
        let licence = geomodel_license();
        let notice = side_install_notice(LicensedAsset {
            name: "BirdNET Geomodel v3.0.2",
            vendor: "Cornell Lab of Ornithology",
            version: "3.0.2",
            license: &licence,
        });

        assert_eq!(
            notice,
            "Also installing BirdNET Geomodel v3.0.2 (Cornell Lab of Ornithology), which is \
             licensed separately: CC-BY-SA-4.0 (share-alike).\n\
             Terms: https://creativecommons.org/licenses/by-sa/4.0/\n"
        );
    }

    #[test]
    fn test_license_details_converts_to_the_structured_shape() {
        let details = crate::output::LicenseDetails::from(&geomodel_license());

        assert_eq!(details.r#type, "CC-BY-SA-4.0");
        assert!(details.commercial_use);
        assert!(details.attribution_required);
        assert!(details.share_alike);
    }
}
