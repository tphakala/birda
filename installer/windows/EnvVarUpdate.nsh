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
