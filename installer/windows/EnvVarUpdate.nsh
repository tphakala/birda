; EnvVarUpdate.nsh - Simplified PATH manipulation for NSIS
; Based on: https://nsis.sourceforge.io/Environmental_Variables:_append,_prepend,_and_remove_entries
; Simplified to only support HKLM (all users) for reliability

!ifndef _EnvVarUpdate_nsh
!define _EnvVarUpdate_nsh

!include "LogicLib.nsh"
!include "WinMessages.nsh"

; Installer version
!define EnvVarUpdate "!insertmacro EnvVarUpdate"

!macro EnvVarUpdate ResultVar EnvVarName Action PathComponent
  Push "${PathComponent}"
  Push "${Action}"
  Push "${EnvVarName}"
  Call EnvVarUpdate
  Pop "${ResultVar}"
!macroend

Function EnvVarUpdate
  Exch $0 ; EnvVarName (e.g., "PATH")
  Exch
  Exch $1 ; Action (A=Add, R=Remove)
  Exch 2
  Exch $2 ; PathComponent to add/remove
  Exch 2

  Push $3 ; Current PATH value
  Push $4 ; Search result
  Push $5 ; Temp for string manipulation
  Push $6 ; New PATH value

  ; Read current PATH from HKLM (all users)
  ReadRegStr $3 HKLM "SYSTEM\CurrentControlSet\Control\Session Manager\Environment" $0

  ${If} $1 == "A" ; Add to PATH
    ; Check if already exists
    Push "$3;"
    Push "$2;"
    Call StrStr
    Pop $4

    ${If} $4 == ""
      ; Not found, add it
      ${If} $3 == ""
        StrCpy $6 "$2"
      ${Else}
        StrCpy $6 "$3;$2"
      ${EndIf}

      ; Write updated PATH
      WriteRegExpandStr HKLM "SYSTEM\CurrentControlSet\Control\Session Manager\Environment" $0 $6
      StrCpy $0 "1" ; Success
    ${Else}
      StrCpy $0 "0" ; Already exists
    ${EndIf}

  ${ElseIf} $1 == "R" ; Remove from PATH
    ; Remove all occurrences
    Push $3
    Push "$2;"
    Push ""
    Call StrRep
    Pop $6

    Push $6
    Push ";$2"
    Push ""
    Call StrRep
    Pop $6

    Push $6
    Push "$2"
    Push ""
    Call StrRep
    Pop $6

    ; Clean up multiple semicolons
    Push $6
    Push ";;"
    Push ";"
    Call StrRep
    Pop $6

    ; Write updated PATH
    WriteRegExpandStr HKLM "SYSTEM\CurrentControlSet\Control\Session Manager\Environment" $0 $6
    StrCpy $0 "1" ; Success
  ${Else}
    StrCpy $0 "0" ; Unknown action
  ${EndIf}

  ; Broadcast environment change
  SendMessage ${HWND_BROADCAST} ${WM_WININICHANGE} 0 "STR:Environment" /TIMEOUT=5000

  Pop $6
  Pop $5
  Pop $4
  Pop $3
  Pop $2
  Pop $1
  Exch $0 ; Return value
FunctionEnd

; String replacement function
!macro StrRep Output Input Find Replace
  Push "${Input}"
  Push "${Find}"
  Push "${Replace}"
  Call StrRep
  Pop "${Output}"
!macroend

Function StrRep
  Exch $R0 ; Replace
  Exch
  Exch $R1 ; Find
  Exch 2
  Exch $R2 ; Input
  Push $R3
  Push $R4
  Push $R5
  Push $R6
  Push $R7
  Push $R8

  StrCpy $R3 ""
  StrCpy $R4 0
  StrLen $R5 $R1

  ${If} $R5 == 0
    StrCpy $R0 $R2
    Goto StrRep_Done
  ${EndIf}

  StrRep_Loop:
    StrCpy $R6 $R2 $R5 $R4
    ${If} $R6 == $R1
      StrCpy $R8 $R2 $R4
      StrCpy $R3 "$R3$R8$R0"
      IntOp $R4 $R4 + $R5
    ${Else}
      StrCpy $R8 $R2 1 $R4
      StrCpy $R3 "$R3$R8"
      IntOp $R4 $R4 + 1
    ${EndIf}

    StrLen $R7 $R2
    ${If} $R4 >= $R7
      StrCpy $R0 $R3
      Goto StrRep_Done
    ${EndIf}
    Goto StrRep_Loop

  StrRep_Done:
  Pop $R8
  Pop $R7
  Pop $R6
  Pop $R5
  Pop $R4
  Pop $R3
  Pop $R2
  Pop $R1
  Exch $R0
FunctionEnd

; String search function
Function StrStr
  Exch $R0 ; Needle
  Exch
  Exch $R1 ; Haystack
  Push $R2
  Push $R3
  Push $R4
  Push $R5

  StrLen $R2 $R0
  StrCpy $R3 0

  StrStr_Loop:
    StrCpy $R4 $R1 $R2 $R3
    ${If} $R4 == ""
      StrCpy $R0 ""
      Goto StrStr_Done
    ${EndIf}
    ${If} $R4 == $R0
      StrCpy $R0 $R4
      Goto StrStr_Done
    ${EndIf}
    IntOp $R3 $R3 + 1
    Goto StrStr_Loop

  StrStr_Done:
  Pop $R5
  Pop $R4
  Pop $R3
  Pop $R2
  Pop $R1
  Exch $R0
FunctionEnd

; Uninstaller version
!define un.EnvVarUpdate "!insertmacro un.EnvVarUpdate"

!macro un.EnvVarUpdate ResultVar EnvVarName Action PathComponent
  Push "${PathComponent}"
  Push "${Action}"
  Push "${EnvVarName}"
  Call un.EnvVarUpdate
  Pop "${ResultVar}"
!macroend

Function un.EnvVarUpdate
  ; Same implementation as installer version
  Exch $0
  Exch
  Exch $1
  Exch 2
  Exch $2
  Exch 2

  Push $3
  Push $4
  Push $5
  Push $6

  ReadRegStr $3 HKLM "SYSTEM\CurrentControlSet\Control\Session Manager\Environment" $0

  ${If} $1 == "R"
    Push $3
    Push "$2;"
    Push ""
    Call un.StrRep
    Pop $6

    Push $6
    Push ";$2"
    Push ""
    Call un.StrRep
    Pop $6

    Push $6
    Push "$2"
    Push ""
    Call un.StrRep
    Pop $6

    Push $6
    Push ";;"
    Push ";"
    Call un.StrRep
    Pop $6

    WriteRegExpandStr HKLM "SYSTEM\CurrentControlSet\Control\Session Manager\Environment" $0 $6
    StrCpy $0 "1"
  ${Else}
    StrCpy $0 "0"
  ${EndIf}

  SendMessage ${HWND_BROADCAST} ${WM_WININICHANGE} 0 "STR:Environment" /TIMEOUT=5000

  Pop $6
  Pop $5
  Pop $4
  Pop $3
  Pop $2
  Pop $1
  Exch $0
FunctionEnd

; Uninstaller string replacement function
Function un.StrRep
  Exch $R0 ; Replace
  Exch
  Exch $R1 ; Find
  Exch 2
  Exch $R2 ; Input
  Push $R3
  Push $R4
  Push $R5
  Push $R6
  Push $R7
  Push $R8

  StrCpy $R3 ""
  StrCpy $R4 0
  StrLen $R5 $R1

  ${If} $R5 == 0
    StrCpy $R0 $R2
    Goto un_StrRep_Done
  ${EndIf}

  un_StrRep_Loop:
    StrCpy $R6 $R2 $R5 $R4
    ${If} $R6 == $R1
      StrCpy $R8 $R2 $R4
      StrCpy $R3 "$R3$R8$R0"
      IntOp $R4 $R4 + $R5
    ${Else}
      StrCpy $R8 $R2 1 $R4
      StrCpy $R3 "$R3$R8"
      IntOp $R4 $R4 + 1
    ${EndIf}

    StrLen $R7 $R2
    ${If} $R4 >= $R7
      StrCpy $R0 $R3
      Goto un_StrRep_Done
    ${EndIf}
    Goto un_StrRep_Loop

  un_StrRep_Done:
  Pop $R8
  Pop $R7
  Pop $R6
  Pop $R5
  Pop $R4
  Pop $R3
  Pop $R2
  Pop $R1
  Exch $R0
FunctionEnd

!endif ; _EnvVarUpdate_nsh
