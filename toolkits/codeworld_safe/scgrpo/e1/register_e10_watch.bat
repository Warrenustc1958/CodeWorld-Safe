@echo off
rem Register the E10-O local watchdog: every 5 minutes, hidden, as the current user.
rem Remove with: schtasks /Delete /TN CWS_E10_Watch /F
schtasks /Create /TN CWS_E10_Watch /SC MINUTE /MO 5 /TR "wscript.exe \"C:\Users\26692\cws_diag\e10_local_watch.vbs\"" /F
schtasks /Query /TN CWS_E10_Watch /FO LIST | findstr /I "TaskName Next Status"
