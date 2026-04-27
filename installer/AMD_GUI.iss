; AMD Detection System — Inno Setup 6 installer script
;
; Build manually (from the project root after running scripts/package.py --installer):
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" ^
;       /DStagingDir="%CD%\dist\staging" ^
;       /DOutputDir="%CD%\dist" ^
;       installer\AMD_GUI.iss
;
; The GitHub Actions workflow (build-installer.yml) calls this automatically.

; ── Compile-time defaults ────────────────────────────────────────────────────
#ifndef StagingDir
  #define StagingDir "..\dist\staging"
#endif

#ifndef OutputDir
  #define OutputDir "..\dist"
#endif

#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif

; ── Setup section ────────────────────────────────────────────────────────────
[Setup]
; A unique AppId prevents the installer from being confused with other apps.
AppId={{F2A4C1B8-3D9E-4F72-BC56-7A8E0D1C23F4}
AppName=AMD Detection System
AppVersion={#AppVersion}
AppVerName=AMD Detection System {#AppVersion}
AppPublisher=shield44-project
AppPublisherURL=https://github.com/shield44-project/Age-Related-Macular-Degeneration
AppSupportURL=https://github.com/shield44-project/Age-Related-Macular-Degeneration/issues
AppUpdatesURL=https://github.com/shield44-project/Age-Related-Macular-Degeneration/releases
DefaultDirName={autopf}\AMD Detection System
DefaultGroupName=AMD Detection System
AllowNoIcons=yes
OutputDir={#OutputDir}
OutputBaseFilename=AMD_GUI-{#AppVersion}-Windows-Installer
Compression=lzma2
SolidCompression=yes
; 64-bit only (matches the Qt/MSVC x64 build)
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
DisableProgramGroupPage=auto
; Don't require admin rights if the user installs under their own profile.
PrivilegesRequiredOverridesAllowed=commandline dialog

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; \
  GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

; ── Files ────────────────────────────────────────────────────────────────────
[Files]
; Qt C++ GUI executable
Source: "{#StagingDir}\bin\AMD_GUI.exe"; DestDir: "{app}\bin"; \
  Flags: ignoreversion

; Qt 6 runtime DLLs (windeployqt copies these beside AMD_GUI.exe)
Source: "{#StagingDir}\bin\*.dll"; DestDir: "{app}\bin"; \
  Flags: ignoreversion skipifsourcedoesntexist

; Qt platform plugin (required — qwindows.dll etc.)
Source: "{#StagingDir}\bin\platforms\*"; DestDir: "{app}\bin\platforms"; \
  Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist

; Qt image format plugins
Source: "{#StagingDir}\bin\imageformats\*"; DestDir: "{app}\bin\imageformats"; \
  Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist

; Qt icon engine plugins
Source: "{#StagingDir}\bin\iconengines\*"; DestDir: "{app}\bin\iconengines"; \
  Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist

; Qt widget style plugins
Source: "{#StagingDir}\bin\styles\*"; DestDir: "{app}\bin\styles"; \
  Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist

; Frozen Python backend (produced by PyInstaller — optional)
; When present the GUI launches it directly without needing Python installed.
Source: "{#StagingDir}\bin\backend_server.exe"; DestDir: "{app}\bin"; \
  Flags: ignoreversion skipifsourcedoesntexist

; Python backend source files (fallback when backend_server.exe is absent)
Source: "{#StagingDir}\backend\*"; DestDir: "{app}\backend"; \
  Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist

; Python requirements (for manual venv setup)
Source: "{#StagingDir}\requirements.txt"; DestDir: "{app}"; \
  Flags: ignoreversion skipifsourcedoesntexist

; ── Shortcuts ────────────────────────────────────────────────────────────────
[Icons]
Name: "{group}\AMD Detection System"; Filename: "{app}\bin\AMD_GUI.exe"
Name: "{group}\{cm:UninstallProgram,AMD Detection System}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\AMD Detection System"; Filename: "{app}\bin\AMD_GUI.exe"; \
  Tasks: desktopicon

; ── Post-install launch ───────────────────────────────────────────────────────
[Run]
Filename: "{app}\bin\AMD_GUI.exe"; \
  Description: "{cm:LaunchProgram,AMD Detection System}"; \
  Flags: nowait postinstall skipifsilent
