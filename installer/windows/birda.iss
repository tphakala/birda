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
; Compression
Compression=lzma2/ultra64
SolidCompression=yes
; Require admin for Program Files installation
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog
; Modern installer look
WizardStyle=modern
; License
LicenseFile=..\..\LICENSE
; Uninstaller
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "addtopath"; Description: "Add to PATH environment variable"; GroupDescription: "Additional options:"

[Files]
; Main executable (dist is in repo root, not installer/windows/)
Source: "..\..\dist\birda.exe"; DestDir: "{app}"; Flags: ignoreversion

; ONNX Runtime main library must be next to executable (for load-dynamic)
Source: "..\..\dist\onnxruntime.dll"; DestDir: "{app}"; Flags: ignoreversion

; Other GPU libraries in lib subdirectory
Source: "..\..\dist\onnxruntime_providers_*.dll"; DestDir: "{app}\lib"; Flags: ignoreversion
Source: "..\..\dist\cudart64_*.dll"; DestDir: "{app}\lib"; Flags: ignoreversion
Source: "..\..\dist\cublas64_*.dll"; DestDir: "{app}\lib"; Flags: ignoreversion
Source: "..\..\dist\cublasLt64_*.dll"; DestDir: "{app}\lib"; Flags: ignoreversion
Source: "..\..\dist\cufft64_*.dll"; DestDir: "{app}\lib"; Flags: ignoreversion
Source: "..\..\dist\cudnn*.dll"; DestDir: "{app}\lib"; Flags: ignoreversion

; Documentation in docs subdirectory
Source: "..\..\dist\README.md"; DestDir: "{app}\docs"; Flags: ignoreversion
Source: "..\..\dist\LICENSE"; DestDir: "{app}\docs"; Flags: ignoreversion skipifsourcedoesntexist
Source: "..\..\dist\THIRD_PARTY_LICENSES.txt"; DestDir: "{app}\docs"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName} Command Prompt"; Filename: "{cmd}"; Parameters: "/k ""{app}\{#MyAppExeName}"" --help"; WorkingDir: "{app}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"

[Registry]
; Add app and lib directories to PATH if user selected that option
Root: HKLM; Subkey: "SYSTEM\CurrentControlSet\Control\Session Manager\Environment"; ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app};{app}\lib"; Tasks: addtopath; Check: NeedsAddPath('{app}')

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
  LibPath: string;
  P: Integer;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    if RegQueryStringValue(HKEY_LOCAL_MACHINE,
      'SYSTEM\CurrentControlSet\Control\Session Manager\Environment',
      'Path', Path) then
    begin
      AppPath := ExpandConstant('{app}');
      LibPath := AppPath + '\lib';

      { Remove lib path first }
      P := Pos(';' + LibPath, Path);
      if P <> 0 then
        Delete(Path, P, Length(';' + LibPath));

      { Remove app path }
      P := Pos(';' + AppPath, Path);
      if P <> 0 then
        Delete(Path, P, Length(';' + AppPath));

      RegWriteStringValue(HKEY_LOCAL_MACHINE,
        'SYSTEM\CurrentControlSet\Control\Session Manager\Environment',
        'Path', Path);
    end;
  end;
end;
