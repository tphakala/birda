# macOS Code Signing Fix - v1.6.0 Release

## Problem Summary

The v1.6.0 release workflow failed at the macOS code signing step with the error:

```
Developer ID Application: ***: no identity found
```

## Root Cause

**Location:** `.github/workflows/release.yml:290`

The workflow was constructing the certificate identity string incorrectly:

```yaml
codesign --sign "Developer ID Application: ${{ secrets.APPLE_TEAM_ID }}"
```

### What Went Wrong

- The `APPLE_TEAM_ID` secret contains only the Team ID (format: `ABC123XYZ`)
- The workflow was prepending "Developer ID Application: " to create `"Developer ID Application: ABC123XYZ"`
- But the actual certificate name in the keychain is `"Developer ID Application: Your Name (ABC123XYZ)"`
- Codesign couldn't find a match, resulting in "no identity found"

### Certificate Import Was Successful

The certificate **was successfully imported** into the keychain (verified by logs showing "Imported Private Key"). The problem was purely in how we were referencing it.

## The Fix

Added a new step to automatically discover the certificate's actual name from the keychain:

```yaml
- name: Find certificate identity
  run: |
    echo "=== Listing available signing identities ==="
    security find-identity -v -p codesigning signing_temp.keychain

    # Extract the full certificate name (between quotes)
    IDENTITY=$(security find-identity -v -p codesigning signing_temp.keychain | grep "Developer ID Application" | head -1 | awk -F'"' '{print $2}')

    if [ -z "$IDENTITY" ]; then
      echo "ERROR: No Developer ID Application certificate found!"
      exit 1
    fi

    echo "Found identity: $IDENTITY"
    echo "CERT_IDENTITY=$IDENTITY" >> $GITHUB_ENV
```

Then updated all `codesign` commands to use the discovered identity:

```yaml
codesign --sign "$CERT_IDENTITY" \
```

## Changes Made

### Files Modified
- `.github/workflows/release.yml`

### Steps Updated
1. **New step:** "Find certificate identity" - Discovers the actual certificate name
2. **Updated:** "Sign the binary" - Uses `$CERT_IDENTITY` instead of hardcoded string
3. **Updated:** "Sign ONNX Runtime library" - Uses `$CERT_IDENTITY`
4. **Updated:** "Sign DMG" - Uses `$CERT_IDENTITY`

### Not Changed
- `notarytool` commands still use `--team-id "${{ secrets.APPLE_TEAM_ID }}"` - This is correct!
- The notarytool command expects the raw Team ID, not the certificate name.

## GitHub Secrets Configuration

The following secrets are required and appear to be correctly configured:

| Secret Name | Purpose | Format |
|-------------|---------|--------|
| `APPLE_CERTIFICATE_BASE64` | Base64-encoded .p12 certificate file | Base64 string |
| `APPLE_CERTIFICATE_PASSWORD` | Password for the .p12 certificate | Plain text |
| `APPLE_TEAM_ID` | Apple Developer Team ID | 10-character alphanumeric (e.g., `ABC123XYZ`) |
| `APPLE_ID` | Apple ID email for notarization | Email address |
| `APPLE_APP_SPECIFIC_PASSWORD` | App-specific password for notarization | Generated password |

### How to Get These Values

1. **APPLE_CERTIFICATE_BASE64**:
   ```bash
   base64 -i YourCertificate.p12 -o certificate.txt
   ```

2. **APPLE_CERTIFICATE_PASSWORD**: The password you set when exporting the certificate from Keychain Access

3. **APPLE_TEAM_ID**: Found in Apple Developer Account → Membership details

4. **APPLE_ID**: Your Apple ID email (must be in the Developer team)

5. **APPLE_APP_SPECIFIC_PASSWORD**: Generated at https://appleid.apple.com/account/manage → App-Specific Passwords

## Testing the Fix

### Option 1: Re-run the Failed Workflow

The easiest way to test:

```bash
gh run rerun 22037479561 --failed
```

This will re-run only the failed "Sign & Notarize macOS" job.

### Option 2: Create a New Test Tag

Create a test release to verify the entire workflow:

```bash
git tag -a v1.6.0-test -m "Test macOS signing fix"
git push origin v1.6.0-test
gh run watch
```

Then delete the test tag and release when confirmed working:

```bash
gh release delete v1.6.0-test --yes
git tag -d v1.6.0-test
git push origin :refs/tags/v1.6.0-test
```

### Option 3: Trigger Manual Workflow Run

If the workflow has `workflow_dispatch` enabled, you can trigger it manually.

## Expected Workflow Output

After the fix, you should see:

```
=== Listing available signing identities ===
  1) ABC123... "Developer ID Application: Your Name (ABC123XYZ)" (CSSMERR_TP_CERT_REVOKED)
Found identity: Developer ID Application: Your Name (ABC123XYZ)
=== Signing binary ===
Using identity: Developer ID Application: Your Name (ABC123XYZ)
unsigned/birda: signed bundle with Mach-O thin (arm64) [com.example.birda]
```

## Next Steps

1. **Commit and push the fix:**
   ```bash
   git add .github/workflows/release.yml
   git commit -m "fix: correct macOS code signing identity detection"
   git push origin main
   ```

2. **Re-run the v1.6.0 release workflow:**
   ```bash
   gh run rerun 22037479561 --failed
   ```

3. **Monitor the workflow:**
   ```bash
   gh run watch
   ```

4. **If successful**, the release will be created automatically with all artifacts including the signed macOS DMG.

## Additional Notes

- The fix is backward compatible and won't affect other jobs
- The discovery method is more robust than hardcoding the identity
- If multiple Developer ID certificates exist, it uses the first one found
- The `APPLE_TEAM_ID` secret is still needed for the notarytool commands

## Troubleshooting

If the workflow still fails:

1. **Check certificate validity:**
   ```bash
   # Locally test the certificate
   openssl pkcs12 -info -in YourCertificate.p12
   ```

2. **Verify the certificate is a "Developer ID Application" certificate**, not:
   - Developer ID Installer
   - Mac App Distribution
   - Mac Installer Distribution

3. **Check certificate expiration** in Apple Developer Console

4. **Ensure the certificate includes the private key** when exported from Keychain Access
