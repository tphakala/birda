//! Small string helpers.

/// Strip `prefix` from the start of `text`, comparing ASCII letters without
/// regard to case.
///
/// Returns `None` when `text` does not start with `prefix`, including when the
/// prefix length would end inside a multi-byte character of `text`.
///
/// # Examples
///
/// ```
/// use birda::utils::text::strip_prefix_ignore_ascii_case;
///
/// assert_eq!(
///     strip_prefix_ignore_ascii_case("OpenVINO_2026", "openvino"),
///     Some("_2026")
/// );
/// assert_eq!(strip_prefix_ignore_ascii_case("other", "openvino"), None);
/// ```
#[must_use]
pub fn strip_prefix_ignore_ascii_case<'a>(text: &'a str, prefix: &str) -> Option<&'a str> {
    let (head, rest) = text.split_at_checked(prefix.len())?;
    head.eq_ignore_ascii_case(prefix).then_some(rest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_matching_ignores_ascii_case_and_never_splits_a_character() {
        assert_eq!(
            strip_prefix_ignore_ascii_case("OpenVINO_2026", "openvino"),
            Some("_2026")
        );
        assert_eq!(
            strip_prefix_ignore_ascii_case("openvino", "OPENVINO"),
            Some("")
        );
        assert_eq!(strip_prefix_ignore_ascii_case("openvin", "openvino"), None);
        assert_eq!(
            strip_prefix_ignore_ascii_case("other_2026", "openvino"),
            None
        );
        // The prefix length (1) lands inside the two-byte "é".
        assert_eq!(strip_prefix_ignore_ascii_case("é", "x"), None);
    }
}
