' Hidden launcher for the E10-O local watchdog (run by Task Scheduler every 5 minutes)
CreateObject("WScript.Shell").Run """C:\Program Files\Git\bin\bash.exe"" ""C:/Users/26692/cws_diag/e10_local_watch.sh""", 0, True
