; Birda Windows Installer Script for Inno Setup
; https://jrsoftware.org/isinfo.php

#define MyAppName "Birda"
#define MyAppPublisher "Tomi P. Hakala"
#define MyAppPublisherURL "https://github.com/tphakala"
#define MyAppURL "https://github.com/tphakala/birda"
#define MyAppExeName "birda.exe"

[Setup]
; NOTE: The value of AppId uniquely identifies this application.
AppId={{B1RDA-GPU-4N4LYZ3R-2024}
AppName={#MyAppName}
; 64-bit application - install to Program Files, not Program Files (x86)
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppPublisherURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
; Output settings (overridden by command line)
OutputDir=.
OutputBaseFilename=birda-windows-x64-cuda-setup
; Compression (using fast for quicker builds, files are already compressed)
Compression=lzma2/fast
SolidCompression=yes
; Require admin for Program Files installation and VC++ Redistributable
PrivilegesRequired=admin
; Modern installer look
WizardStyle=modern
; License (combined license with third-party notices)
LicenseFile=INSTALLER_LICENSE.txt
; Info page after installation (TensorRT instructions)
InfoAfterFile=TENSORRT_INFO.txt
; Uninstaller
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}
; Broadcast environment variable changes (for PATH)
ChangesEnvironment=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "addtopath"; Description: "Add to PATH environment variable"; GroupDescription: "Additional options:"

[Files]
; Main executable (dist is in repo root, not installer/windows/)
Source: "..\..\dist\birda.exe"; DestDir: "{app}"; Flags: ignoreversion

; ONNX Runtime main library must be next to executable (for load-dynamic)
Source: "..\..\dist\onnxruntime.dll"; DestDir: "{app}"; Flags: ignoreversion

; GPU libraries next to executable (Windows DLL search order)
Source: "..\..\dist\onnxruntime_providers_*.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\dist\cudart64_*.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\dist\cublas64_*.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\dist\cublasLt64_*.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\dist\cufft64_*.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\dist\cudnn*.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\dist\nvrtc*.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\dist\nvJitLink*.dll"; DestDir: "{app}"; Flags: ignoreversion
; Note: TensorRT libraries are NOT bundled due to size constraints
; Users can optionally download TensorRT from https://github.com/NVIDIA/TensorRT

; Documentation in docs subdirectory
Source: "..\..\dist\README.md"; DestDir: "{app}\docs"; Flags: ignoreversion
Source: "..\..\dist\LICENSE"; DestDir: "{app}\docs"; Flags: ignoreversion skipifsourcedoesntexist
Source: "..\..\dist\THIRD_PARTY_LICENSES.txt"; DestDir: "{app}\docs"; Flags: ignoreversion

; Visual C++ Redistributable (required by onnxruntime.dll)
; Note: Always include in installer, check at runtime whether to install
Source: "..\..\dist\vc_redist.x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

[Run]
; Install VC++ Redistributable (installer will skip if up-to-date version exists)
Filename: "{tmp}\vc_redist.x64.exe"; Parameters: "/quiet /norestart"; StatusMsg: "Installing Visual C++ Runtime..."; Flags: waituntilterminated
; Post-install option to download Birda GUI
Filename: "https://github.com/tphakala/birda-gui/releases/latest"; Description: "Download Birda GUI (optional graphical interface)"; Flags: postinstall shellexec skipifsilent unchecked

[Icons]
Name: "{group}\{#MyAppName} Command Prompt"; Filename: "{cmd}"; Parameters: "/k ""{app}\{#MyAppExeName}"" --help"; WorkingDir: "{app}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"

[Registry]
; Add app directory to PATH if user selected that option
Root: HKLM; Subkey: "SYSTEM\CurrentControlSet\Control\Session Manager\Environment"; ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; Tasks: addtopath; Check: NeedsAddPath('{app}')

[Code]
function NeedsAddPath(Param: string): boolean;
var
  OrigPath: string;
begin
  if not RegQueryStringValue(HKEY_LOCAL_MACHINE,
    'SYSTEM\CurrentControlSet\Control\Session Manager\Environment',
    'Path', OrigPath)
  then begin
    Result := True;
    exit;
  end;
  { look for the path with leading and trailing semicolon }
  Result := Pos(';' + Param + ';', ';' + OrigPath + ';') = 0;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Path: string;
  AppPath: string;
  P: Integer;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    if RegQueryStringValue(HKEY_LOCAL_MACHINE,
      'SYSTEM\CurrentControlSet\Control\Session Manager\Environment',
      'Path', Path) then
    begin
      AppPath := ExpandConstant('{app}');

      { Remove app path }
      P := Pos(';' + AppPath, Path);
      if P <> 0 then
        Delete(Path, P, Length(';' + AppPath));

      RegWriteStringValue(HKEY_LOCAL_MACHINE,
        'SYSTEM\CurrentControlSet\Control\Session Manager\Environment',
        'Path', Path);

      // Broadcast environment change
      BroadcastEnvironmentChange();
    end;
  end;
end;
