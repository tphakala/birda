# NSIS Test Installer Design

## Overview

Create a parallel NSIS-based Windows installer to evaluate as an alternative to Inno Setup. This allows direct comparison of stability, maintainability, and robustness between the two installer systems.

## Motivation

Recent Inno Setup issues with GitHub Actions runner updates (duplicate identifier errors for Windows API constants) prompted evaluation of alternatives. NSIS offers:

- More stable scripting environment (assembly-like, less prone to breaking changes)
- Direct Windows API integration without external declarations
- Large community and extensive testing
- Well-documented macro libraries for common tasks

## Goals

1. Create NSIS installer with **full feature parity** to Inno Setup
2. Enable side-by-side comparison without disrupting production releases
3. Evaluate maintainability and stability of NSIS vs Inno Setup

## Non-Goals

- Not replacing Inno Setup immediately
- Not creating a new release type (test only, draft releases)
- Not modifying existing release workflow

## Design

### Workflow Structure

**File:** `.github/workflows/nsis-test.yml`

**Trigger:** Manual dispatch with version input
- User specifies version tag (e.g., `v1.4.2`)
- Downloads pre-built artifacts from that release
- Builds NSIS installer
- Creates draft release for comparison

**Benefits:**
- Full control over when to test
- No interference with production releases
- Reuses existing build artifacts (no rebuild needed)

### NSIS Script Structure

**File:** `installer/windows/birda.nsi`

**Sections:**

1. **General Configuration**
   - Product metadata (name, version, publisher, URLs)
   - Install directory: `$PROGRAMFILES64\Birda`
   - Request admin privileges (`RequestExecutionLevel admin`)
   - LZMA compression for smaller installers

2. **Installer Pages**
   - License page (INSTALLER_LICENSE.txt)
   - Directory selection (customizable install path)
   - Components page (PATH checkbox option)
   - Installation progress
   - Finish page with TensorRT information

3. **Installation Section**
   - Install birda.exe + all CUDA/cuDNN DLLs to `$INSTDIR`
   - Install documentation to `$INSTDIR\docs\`
   - Execute VC++ Redistributable installer (`/quiet /norestart`)
   - Optional: Add to system PATH
   - Broadcast `WM_SETTINGCHANGE` to update environment
   - Create Start Menu shortcuts

4. **Uninstaller Section**
   - Remove all installed files
   - Remove from PATH if added during install
   - Broadcast environment change
   - Remove Start Menu shortcuts
   - Delete uninstaller itself

### Key Technical Decisions

#### PATH Manipulation

Use `EnvVarUpdate` macro from NSIS wiki:
- Handles PATH deduplication
- Thread-safe registry operations
- Proper error handling

#### Environment Broadcasting

Direct Windows API call using NSIS `System` plugin:
```nsis
System::Call 'user32::SendMessageTimeout(i 0xFFFF, i 0x1A, i 0, t "Environment", i 2, i 5000, *i .r0)'
```

Values:
- `0xFFFF` = HWND_BROADCAST
- `0x1A` = WM_SETTINGCHANGE
- `2` = SMTO_ABORTIFHUNG
- `5000` = 5 second timeout

**Note:** These are literal hex values, no constant declarations needed (avoids Inno Setup duplicate identifier issue).

#### VC++ Redistributable

Use `ExecWait` with return code checking:
```nsis
ExecWait '"$INSTDIR\vc_redist.x64.exe" /quiet /norestart' $0
DetailPrint "VC++ Redistributable installer returned: $0"
```

### Artifact Reuse Strategy

**Download from existing release:**
1. Use `gh release download {version}` to get `birda-windows-x64-cuda-{version}.zip`
2. Extract to `dist/` directory
3. Build NSIS installer from extracted files

**Alternative considered:** Download from workflow artifacts
- More complex (needs workflow run ID)
- Doesn't match "test existing release" use case

### Output and Testing

**Produced artifacts:**
- `birda-windows-x64-cuda-nsis-{version}-setup.exe`
- Draft release: "NSIS Test Build {version}"

**Release description includes:**
- Installation instructions
- Comparison checklist:
  - Installer size (MB)
  - Build time (seconds)
  - Manual test results (installation, PATH, uninstall)
- Link to design document

## Implementation Plan

1. Create NSIS script (`installer/windows/birda.nsi`)
2. Create workflow file (`.github/workflows/nsis-test.yml`)
3. Create supporting documentation files if missing
4. Test manually via workflow dispatch

## Success Criteria

- NSIS installer builds successfully on GitHub Actions
- Installer has feature parity with Inno Setup
- Manual testing confirms:
  - Installation completes without errors
  - PATH configuration works
  - VC++ Redistributable installs
  - Uninstaller removes everything cleanly
- Draft release created with comparison data

## Future Considerations

If NSIS proves superior:
1. Gradually migrate release workflow to NSIS
2. Keep Inno Setup script for one release cycle (parallel releases)
3. Deprecate Inno Setup after successful NSIS adoption
4. Update documentation

## References

- [NSIS Documentation](https://nsis.sourceforge.io/Docs/)
- [EnvVarUpdate Macro](https://nsis.sourceforge.io/Environmental_Variables:_append,_prepend,_and_remove_entries)
- [System Plugin](https://nsis.sourceforge.io/System_plug-in)
- Inno Setup script: `installer/windows/birda.iss`
