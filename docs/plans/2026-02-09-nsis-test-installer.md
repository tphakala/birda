# NSIS Test Installer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create NSIS-based Windows installer with full feature parity to Inno Setup for comparison testing.

**Architecture:** Manual-dispatch GitHub Actions workflow downloads pre-built release artifacts, builds NSIS installer, and creates draft release for side-by-side comparison with Inno Setup.

**Tech Stack:** NSIS 3.x, GitHub Actions, PowerShell, Windows API (SendMessageTimeout)

---

## Task 1: Create NSIS Installer Script

**Files:**
- Create: `installer/windows/birda.nsi`
- Reference: `installer/windows/birda.iss` (existing Inno Setup script)

**Step 1: Create base NSIS script with metadata**

Create `installer/windows/birda.nsi`:

```nsis
; Birda Windows Installer Script for NSIS
; https://nsis.sourceforge.io/

!define PRODUCT_NAME "Birda"
!define PRODUCT_VERSION "1.0.0" ; Will be overridden by makensis /D flag
!define PRODUCT_PUBLISHER "Tomi P. Hakala"
!define PRODUCT_WEB_SITE "https://github.com/tphakala/birda"
!define PRODUCT_DIR_REGKEY "Software\Microsoft\Windows\CurrentVersion\App Paths\birda.exe"
!define PRODUCT_UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}"
!define PRODUCT_UNINST_ROOT_KEY "HKLM"

; Modern UI
!include "MUI2.nsh"

; Environment variable manipulation
!include "EnvVarUpdate.nsh"

; Request admin privileges
RequestExecutionLevel admin

; Installer name and output file
Name "${PRODUCT_NAME} ${PRODUCT_VERSION}"
OutFile "birda-windows-x64-cuda-nsis-setup.exe"

; Default installation directory
InstallDir "$PROGRAMFILES64\${PRODUCT_NAME}"
InstallDirRegKey HKLM "${PRODUCT_DIR_REGKEY}" ""

; Show installation details
ShowInstDetails show
ShowUnInstDetails show

; Compression
SetCompressor /SOLID lzma
```

**Step 2: Add Modern UI configuration**

Add to `installer/windows/birda.nsi`:

```nsis
; MUI Settings
!define MUI_ABORTWARNING
!define MUI_ICON "${NSISDIR}\Contrib\Graphics\Icons\modern-install.ico"
!define MUI_UNICON "${NSISDIR}\Contrib\Graphics\Icons\modern-uninstall.ico"

; Welcome page
!insertmacro MUI_PAGE_WELCOME

; License page
!insertmacro MUI_PAGE_LICENSE "..\..\installer\windows\INSTALLER_LICENSE.txt"

; Directory page
!insertmacro MUI_PAGE_DIRECTORY

; Components page (for PATH checkbox)
!insertmacro MUI_PAGE_COMPONENTS

; Installation page
!insertmacro MUI_PAGE_INSTFILES

; Finish page with TensorRT info
!define MUI_FINISHPAGE_SHOWREADME "$INSTDIR\docs\TENSORRT_INFO.txt"
!define MUI_FINISHPAGE_SHOWREADME_TEXT "View TensorRT installation instructions"
!define MUI_FINISHPAGE_LINK "Download Birda GUI (optional graphical interface)"
!define MUI_FINISHPAGE_LINK_LOCATION "https://github.com/tphakala/birda-gui/releases/latest"
!insertmacro MUI_PAGE_FINISH

; Uninstaller pages
!insertmacro MUI_UNPAGE_INSTFILES

; Language
!insertmacro MUI_LANGUAGE "English"
```

**Step 3: Add main installation section**

Add to `installer/windows/birda.nsi`:

```nsis
; Main installation section
Section "Birda (required)" SEC01
  SectionIn RO ; Read-only (always installed)

  SetOutPath "$INSTDIR"
  SetOverwrite ifnewer

  ; Copy executable
  File "..\..\dist\birda.exe"

  ; Copy all DLLs
  File "..\..\dist\*.dll"

  ; Create docs subdirectory
  CreateDirectory "$INSTDIR\docs"
  SetOutPath "$INSTDIR\docs"

  ; Copy documentation
  File "..\..\dist\README.md"
  File /nonfatal "..\..\dist\LICENSE"
  File "..\..\dist\THIRD_PARTY_LICENSES.txt"

  ; Copy info files for installer
  File "..\..\installer\windows\TENSORRT_INFO.txt"

  ; Create shortcuts
  CreateDirectory "$SMPROGRAMS\${PRODUCT_NAME}"
  CreateShortCut "$SMPROGRAMS\${PRODUCT_NAME}\Birda Command Prompt.lnk" "$WINDIR\System32\cmd.exe" '/k "$INSTDIR\birda.exe" --help' "$INSTDIR\birda.exe" 0
  CreateShortCut "$SMPROGRAMS\${PRODUCT_NAME}\Uninstall.lnk" "$INSTDIR\uninst.exe"
SectionEnd

; VC++ Redistributable installation
Section "Visual C++ Runtime" SEC02
  SectionIn RO ; Always install

  SetOutPath "$TEMP"
  File "..\..\dist\vc_redist.x64.exe"

  DetailPrint "Installing Visual C++ Redistributable..."
  ExecWait '"$TEMP\vc_redist.x64.exe" /quiet /norestart' $0
  DetailPrint "VC++ Redistributable installer returned: $0"

  Delete "$TEMP\vc_redist.x64.exe"
SectionEnd
```

**Step 4: Add PATH configuration section**

Add to `installer/windows/birda.nsi`:

```nsis
; Optional PATH configuration
Section "Add to PATH" SEC03
  ; Add installation directory to system PATH
  ${EnvVarUpdate} $0 "PATH" "A" "HKLM" "$INSTDIR"

  ; Broadcast environment change
  DetailPrint "Broadcasting environment change..."
  System::Call 'user32::SendMessageTimeout(i 0xFFFF, i 0x1A, i 0, t "Environment", i 2, i 5000, *i .r0) i .r1'
  DetailPrint "SendMessageTimeout returned: $1"
SectionEnd

; Section descriptions
!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC01} "Birda executable and required libraries"
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC02} "Visual C++ Runtime required by ONNX Runtime"
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC03} "Add Birda to system PATH (recommended)"
!insertmacro MUI_FUNCTION_DESCRIPTION_END
```

**Step 5: Add post-installation section**

Add to `installer/windows/birda.nsi`:

```nsis
Section -Post
  WriteUninstaller "$INSTDIR\uninst.exe"
  WriteRegStr HKLM "${PRODUCT_DIR_REGKEY}" "" "$INSTDIR\birda.exe"
  WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "DisplayName" "$(^Name)"
  WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "UninstallString" "$INSTDIR\uninst.exe"
  WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "DisplayIcon" "$INSTDIR\birda.exe"
  WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "DisplayVersion" "${PRODUCT_VERSION}"
  WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "URLInfoAbout" "${PRODUCT_WEB_SITE}"
  WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "Publisher" "${PRODUCT_PUBLISHER}"
SectionEnd
```

**Step 6: Add uninstaller section**

Add to `installer/windows/birda.nsi`:

```nsis
; Uninstaller
Section Uninstall
  ; Remove from PATH if it was added
  ${un.EnvVarUpdate} $0 "PATH" "R" "HKLM" "$INSTDIR"

  ; Broadcast environment change
  System::Call 'user32::SendMessageTimeout(i 0xFFFF, i 0x1A, i 0, t "Environment", i 2, i 5000, *i .r0)'

  ; Remove shortcuts
  Delete "$SMPROGRAMS\${PRODUCT_NAME}\*.*"
  RMDir "$SMPROGRAMS\${PRODUCT_NAME}"

  ; Remove files
  Delete "$INSTDIR\birda.exe"
  Delete "$INSTDIR\*.dll"
  Delete "$INSTDIR\docs\*.*"
  RMDir "$INSTDIR\docs"
  Delete "$INSTDIR\uninst.exe"

  ; Remove directory if empty
  RMDir "$INSTDIR"

  ; Remove registry keys
  DeleteRegKey ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}"
  DeleteRegKey HKLM "${PRODUCT_DIR_REGKEY}"

  SetAutoClose true
SectionEnd
```

**Step 7: Add initialization functions**

Add to `installer/windows/birda.nsi`:

```nsis
; Installer initialization
Function .onInit
  ; Check if running on 64-bit Windows
  ${If} ${RunningX64}
    ; OK
  ${Else}
    MessageBox MB_OK|MB_ICONSTOP "This installer requires 64-bit Windows."
    Abort
  ${EndIf}
FunctionEnd

; Uninstaller initialization
Function un.onInit
  MessageBox MB_ICONQUESTION|MB_YESNO|MB_DEFBUTTON2 "Are you sure you want to uninstall $(^Name)?" IDYES +2
  Abort
FunctionEnd

; Close function
Function un.onUninstSuccess
  HideWindow
  MessageBox MB_ICONINFORMATION|MB_OK "$(^Name) was successfully removed from your computer."
FunctionEnd
```

**Step 8: Download EnvVarUpdate.nsh macro**

Create `installer/windows/EnvVarUpdate.nsh` with the standard EnvVarUpdate macro from NSIS wiki:

```nsis
; EnvVarUpdate.nsh
; Environment variable update macro
; From: https://nsis.sourceforge.io/Environmental_Variables:_append,_prepend,_and_remove_entries

!ifndef _EnvVarUpdate_nsh
!define _EnvVarUpdate_nsh

!include "LogicLib.nsh"

!define EnvVarUpdate "!insertmacro EnvVarUpdate"

!macro EnvVarUpdate ResultVar EnvVarName Action Regloc PathComponent
  Push "${EnvVarName}"
  Push "${Action}"
  Push "${RegLoc}"
  Push "${PathComponent}"
  Call EnvVarUpdate
  Pop "${ResultVar}"
!macroend

Function EnvVarUpdate
  Exch $0 ; PathComponent
  Exch
  Exch $1 ; RegLoc
  Exch 2
  Exch $2 ; Action
  Exch 2
  Exch $3 ; EnvVarName
  Exch 3

  Push $4
  Push $5
  Push $6

  ; Read current PATH
  ReadRegStr $4 $1 "SYSTEM\CurrentControlSet\Control\Session Manager\Environment" $3

  ; Check if path component already exists
  StrCpy $6 "$4;"
  Push $6
  Push "$0;"
  Call StrStr
  Pop $5

  ${If} $2 == "A" ; Add
    ${If} $5 == ""
      ${If} $4 == ""
        StrCpy $4 "$0"
      ${Else}
        StrCpy $4 "$4;$0"
      ${EndIf}
      WriteRegExpandStr $1 "SYSTEM\CurrentControlSet\Control\Session Manager\Environment" $3 $4
    ${EndIf}
  ${ElseIf} $2 == "R" ; Remove
    ${If} $5 != ""
      Push $4
      Push "$0;"
      Push ""
      Call StrReplace
      Pop $4
      Push $4
      Push ";$0"
      Push ""
      Call StrReplace
      Pop $4
      WriteRegExpandStr $1 "SYSTEM\CurrentControlSet\Control\Session Manager\Environment" $3 $4
    ${EndIf}
  ${EndIf}

  StrCpy $0 $4

  Pop $6
  Pop $5
  Pop $4
  Pop $3
  Pop $2
  Pop $1
  Exch $0
FunctionEnd

; String search function
Function StrStr
  Exch $1 ; Search string
  Exch
  Exch $0 ; String
  Push $2
  Push $3

  StrCpy $2 0
  StrLen $3 $1

  loop:
    StrCpy $4 $0 $3 $2
    StrCmp $4 "" notfound
    StrCmp $4 $1 found
    IntOp $2 $2 + 1
    Goto loop

  found:
    StrCpy $0 $0 $2
    Goto end

  notfound:
    StrCpy $0 ""

  end:
  Pop $4
  Pop $3
  Pop $2
  Pop $1
  Exch $0
FunctionEnd

; String replace function
Function StrReplace
  Exch $2 ; Replacement
  Exch
  Exch $1 ; Search string
  Exch 2
  Exch $0 ; String
  Push $3
  Push $4
  Push $5

  StrCpy $3 0
  StrLen $4 $1

  loop:
    StrCpy $5 $0 $4 $3
    StrCmp $5 "" done
    StrCmp $5 $1 replace
    IntOp $3 $3 + 1
    Goto loop

  replace:
    StrCpy $5 $0 $3
    IntOp $3 $3 + $4
    StrCpy $0 $0 "" $3
    StrCpy $0 "$5$2$0"
    StrLen $3 $5
    IntOp $3 $3 + $2
    Goto loop

  done:
  Pop $5
  Pop $4
  Pop $3
  Pop $2
  Pop $1
  Exch $0
FunctionEnd

; Uninstaller versions
!define un.EnvVarUpdate "!insertmacro un.EnvVarUpdate"

!macro un.EnvVarUpdate ResultVar EnvVarName Action Regloc PathComponent
  Push "${EnvVarName}"
  Push "${Action}"
  Push "${RegLoc}"
  Push "${PathComponent}"
  Call un.EnvVarUpdate
  Pop "${ResultVar}"
!macroend

Function un.EnvVarUpdate
  Exch $0
  Exch
  Exch $1
  Exch 2
  Exch $2
  Exch 2
  Exch $3
  Exch 3

  Push $4
  Push $5
  Push $6

  ReadRegStr $4 $1 "SYSTEM\CurrentControlSet\Control\Session Manager\Environment" $3

  StrCpy $6 "$4;"
  Push $6
  Push "$0;"
  Call un.StrStr
  Pop $5

  ${If} $2 == "R"
    ${If} $5 != ""
      Push $4
      Push "$0;"
      Push ""
      Call un.StrReplace
      Pop $4
      Push $4
      Push ";$0"
      Push ""
      Call un.StrReplace
      Pop $4
      WriteRegExpandStr $1 "SYSTEM\CurrentControlSet\Control\Session Manager\Environment" $3 $4
    ${EndIf}
  ${EndIf}

  StrCpy $0 $4

  Pop $6
  Pop $5
  Pop $4
  Pop $3
  Pop $2
  Pop $1
  Exch $0
FunctionEnd

Function un.StrStr
  Exch $1
  Exch
  Exch $0
  Push $2
  Push $3

  StrCpy $2 0
  StrLen $3 $1

  loop:
    StrCpy $4 $0 $3 $2
    StrCmp $4 "" notfound
    StrCmp $4 $1 found
    IntOp $2 $2 + 1
    Goto loop

  found:
    StrCpy $0 $0 $2
    Goto end

  notfound:
    StrCpy $0 ""

  end:
  Pop $4
  Pop $3
  Pop $2
  Pop $1
  Exch $0
FunctionEnd

Function un.StrReplace
  Exch $2
  Exch
  Exch $1
  Exch 2
  Exch $0
  Push $3
  Push $4
  Push $5

  StrCpy $3 0
  StrLen $4 $1

  loop:
    StrCpy $5 $0 $4 $3
    StrCmp $5 "" done
    StrCmp $5 $1 replace
    IntOp $3 $3 + 1
    Goto loop

  replace:
    StrCpy $5 $0 $3
    IntOp $3 $3 + $4
    StrCpy $0 $0 "" $3
    StrCpy $0 "$5$2$0"
    StrLen $3 $5
    IntOp $3 $3 + $2
    Goto loop

  done:
  Pop $5
  Pop $4
  Pop $3
  Pop $2
  Pop $1
  Exch $0
FunctionEnd

!endif ; _EnvVarUpdate_nsh
```

**Step 9: Commit NSIS script**

```bash
git add installer/windows/birda.nsi installer/windows/EnvVarUpdate.nsh
git commit -m "feat: add NSIS installer script with full feature parity"
```

---

## Task 2: Create GitHub Actions Workflow

**Files:**
- Create: `.github/workflows/nsis-test.yml`

**Step 1: Create workflow with manual dispatch trigger**

Create `.github/workflows/nsis-test.yml`:

```yaml
name: NSIS Test Build

on:
  workflow_dispatch:
    inputs:
      version:
        description: 'Version tag to build installer for (e.g., v1.4.2)'
        required: true
        type: string

env:
  CARGO_TERM_COLOR: always

jobs:
  build-nsis-installer:
    name: Build NSIS Installer
    runs-on: windows-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v6

      - name: Download release artifacts
        shell: pwsh
        run: |
          $version = "${{ github.event.inputs.version }}"
          Write-Host "Downloading artifacts for version: $version"

          # Download GPU Windows build
          gh release download $version --pattern "birda-windows-x64-cuda-$version.zip"

          # Extract to dist directory
          New-Item -ItemType Directory -Force -Path dist
          Expand-Archive -Path "birda-windows-x64-cuda-$version.zip" -DestinationPath "dist" -Force

          Write-Host "Contents of dist/:"
          Get-ChildItem dist -Recurse
        env:
          GH_TOKEN: ${{ github.token }}
```

**Step 2: Add NSIS installation step**

Add to `.github/workflows/nsis-test.yml`:

```yaml
      - name: Install NSIS
        shell: pwsh
        run: |
          # NSIS is pre-installed on Windows runners
          # Verify installation
          $nsisPath = "C:\Program Files (x86)\NSIS\makensis.exe"
          if (Test-Path $nsisPath) {
            Write-Host "NSIS found at: $nsisPath"
            & $nsisPath /VERSION
          } else {
            Write-Host "Installing NSIS via chocolatey..."
            choco install nsis -y
          }
```

**Step 3: Add NSIS build step**

Add to `.github/workflows/nsis-test.yml`:

```yaml
      - name: Build NSIS installer
        shell: pwsh
        run: |
          $version = "${{ github.event.inputs.version }}"
          $versionNumber = $version -replace '^v', ''

          Write-Host "Building NSIS installer for version: $versionNumber"

          # Build with version override
          & "C:\Program Files (x86)\NSIS\makensis.exe" `
            /DPRODUCT_VERSION="$versionNumber" `
            /NOCD `
            "installer\windows\birda.nsi"

          # Rename output file to include version
          Move-Item "birda-windows-x64-cuda-nsis-setup.exe" `
            "birda-windows-x64-cuda-nsis-$version-setup.exe"

          Write-Host "Installer created successfully"
```

**Step 4: Add artifact upload step**

Add to `.github/workflows/nsis-test.yml`:

```yaml
      - name: Upload installer artifact
        uses: actions/upload-artifact@v6
        with:
          name: nsis-installer
          path: birda-windows-x64-cuda-nsis-${{ github.event.inputs.version }}-setup.exe
```

**Step 5: Add draft release creation step**

Add to `.github/workflows/nsis-test.yml`:

```yaml
      - name: Create draft release
        shell: pwsh
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          $version = "${{ github.event.inputs.version }}"
          $installerPath = "birda-windows-x64-cuda-nsis-$version-setup.exe"
          $installerSize = (Get-Item $installerPath).Length / 1MB
          $installerSizeMB = [math]::Round($installerSize, 2)

          # Get Inno Setup installer size for comparison
          gh release download $version --pattern "birda-windows-x64-cuda-$version-setup.exe"
          $innoSize = (Get-Item "birda-windows-x64-cuda-$version-setup.exe").Length / 1MB
          $innoSizeMB = [math]::Round($innoSize, 2)

          $body = @"
          ## NSIS Test Build for $version

          This is a test build using NSIS installer instead of Inno Setup.

          ### Comparison

          | Metric | NSIS | Inno Setup |
          |--------|------|------------|
          | Installer Size | ${installerSizeMB} MB | ${innoSizeMB} MB |

          ### Testing Checklist

          - [ ] Installation completes without errors
          - [ ] Birda executable works correctly
          - [ ] PATH configuration works (if selected)
          - [ ] VC++ Redistributable installs
          - [ ] Start Menu shortcuts created
          - [ ] Uninstaller removes all files
          - [ ] Uninstaller removes from PATH (if added)
          - [ ] Environment variables updated correctly

          ### Installation

          1. Download `$installerPath`
          2. Run as administrator
          3. Follow installation wizard
          4. Optionally add to PATH

          ### Design Document

          See [docs/nsis-test-installer-design.md](../blob/main/docs/nsis-test-installer-design.md) for full design details.
          "@

          # Create draft release
          gh release create "nsis-test-$version" `
            --draft `
            --title "NSIS Test Build $version" `
            --notes $body `
            $installerPath

          Write-Host "Draft release created: nsis-test-$version"
```

**Step 6: Commit workflow**

```bash
git add .github/workflows/nsis-test.yml
git commit -m "feat: add NSIS test workflow for installer comparison"
```

---

## Task 3: Verify and Document

**Files:**
- Modify: `docs/nsis-test-installer-design.md` (add testing instructions)

**Step 1: Add testing instructions to design doc**

Add to end of `docs/nsis-test-installer-design.md`:

```markdown
## Testing Instructions

### Trigger the Workflow

1. Go to GitHub Actions → NSIS Test Build
2. Click "Run workflow"
3. Enter version tag (e.g., `v1.4.2`)
4. Click "Run workflow"

### Verify Build

1. Wait for workflow to complete (~5 minutes)
2. Check workflow logs for errors
3. Download draft release from Releases page
4. Verify installer file exists

### Manual Testing

1. **Installation Test**
   - Run installer as administrator
   - Accept license
   - Choose directory (or use default)
   - Check "Add to PATH" option
   - Complete installation
   - Verify: `birda --version` works in new terminal

2. **PATH Test**
   - Open new PowerShell/CMD window
   - Run: `birda --help`
   - Should work without specifying full path

3. **Uninstaller Test**
   - Run uninstaller from Start Menu or Control Panel
   - Verify all files removed from `C:\Program Files\Birda`
   - Verify Start Menu shortcuts removed
   - Open new terminal and verify `birda` no longer in PATH

### Comparison Notes

Document findings in the draft release:
- Installer stability during build
- Installation experience
- Any errors encountered
- Performance differences
```

**Step 2: Commit documentation update**

```bash
git add docs/nsis-test-installer-design.md
git commit -m "docs: add testing instructions to NSIS design document"
```

**Step 3: Push changes**

```bash
git push origin main
```

---

## Verification

After implementation:

1. **Syntax Check**
   - NSIS script compiles without errors locally (if NSIS installed)
   - Workflow YAML is valid

2. **Manual Workflow Run**
   - Trigger workflow with `v1.4.2` (or latest tag)
   - Verify build completes successfully
   - Verify draft release created

3. **Installer Test** (if possible)
   - Download NSIS installer from draft release
   - Test installation on Windows VM/machine
   - Verify all features work as expected

## Success Criteria

- ✅ NSIS script created with all features from Inno Setup
- ✅ GitHub Actions workflow runs successfully
- ✅ Draft release created with installer
- ✅ Documentation complete with testing instructions
- ✅ All commits follow conventional commit format

---

**Implementation Notes:**

- The EnvVarUpdate macro handles all PATH manipulation edge cases
- Windows API calls use literal hex values (no constant definitions needed)
- Workflow reuses existing release artifacts (no rebuild required)
- Draft releases keep tests separate from production releases
