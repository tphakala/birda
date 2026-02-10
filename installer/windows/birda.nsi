; Birda Windows Installer Script for NSIS
; https://nsis.sourceforge.io/

; Add script directory to include path for /NOCD builds
!addincludedir "${NSISDIR}\Include"
!addincludedir "installer\windows"

!define PRODUCT_NAME "Birda"
!ifndef PRODUCT_VERSION
  !define PRODUCT_VERSION "1.0.0" ; Default if not passed via makensis /D flag
!endif
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

; MUI Settings
!define MUI_ABORTWARNING
!define MUI_ICON "${NSISDIR}\Contrib\Graphics\Icons\modern-install.ico"
!define MUI_UNICON "${NSISDIR}\Contrib\Graphics\Icons\modern-uninstall.ico"

; Welcome page
!insertmacro MUI_PAGE_WELCOME

; License page
!insertmacro MUI_PAGE_LICENSE "installer\windows\INSTALLER_LICENSE.txt"

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

  ; Check exit code: 0=success, 1638=already installed, 3010=reboot required
  ${If} $0 != 0
  ${AndIf} $0 != 1638
  ${AndIf} $0 != 3010
    DetailPrint "WARNING: VC++ Redistributable installation failed with code $0"
    MessageBox MB_OK|MB_ICONEXCLAMATION "Visual C++ Redistributable installation failed (exit code: $0).$\r$\n$\r$\nBirda may not run correctly without it.$\r$\n$\r$\nPlease install it manually from:$\r$\nhttps://aka.ms/vs/17/release/vc_redist.x64.exe"
  ${EndIf}

  ; Always clean up temp file
  Delete "$TEMP\vc_redist.x64.exe"
SectionEnd

; Optional PATH configuration
Section "Add to PATH" SEC03
  ; Add installation directory to system PATH
  ${EnvVarUpdate} $0 "PATH" "A" "$INSTDIR"

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

; Uninstaller
Section Uninstall
  ; Remove from PATH if it was added
  ${un.EnvVarUpdate} $0 "PATH" "R" "$INSTDIR"

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

; Cleanup on installation failure
Function .onInstFailed
  ; Clean up VC++ temp file if it exists
  Delete "$TEMP\vc_redist.x64.exe"
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
