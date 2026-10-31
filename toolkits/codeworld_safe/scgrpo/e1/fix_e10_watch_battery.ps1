# Let the CWS_E10_Watch task run on battery power (default schtasks settings block it), then run it once.
$s = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
      -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 4)
Set-ScheduledTask -TaskName CWS_E10_Watch -Settings $s | Out-Null
$t = Get-ScheduledTask -TaskName CWS_E10_Watch
"DisallowStartIfOnBatteries = $($t.Settings.DisallowStartIfOnBatteries)"
"StopIfGoingOnBatteries     = $($t.Settings.StopIfGoingOnBatteries)"
Start-ScheduledTask -TaskName CWS_E10_Watch
Start-Sleep -Seconds 40
Get-Content C:\Users\26692\cws_diag\e10_local_watch.log -Tail 2
Get-ScheduledTaskInfo -TaskName CWS_E10_Watch | Select-Object LastRunTime, LastTaskResult, NextRunTime
