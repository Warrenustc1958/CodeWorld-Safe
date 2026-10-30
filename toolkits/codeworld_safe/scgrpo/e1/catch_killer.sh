#!/bin/bash
# Read-only: when a process outside Ray/our scripts sends SIGKILL, snapshot who it is (cmdline, cwd, owner,
# parent chain, login env) before it exits. Output: causal/killtrace/killers.log
O=/data_cf/cws_sam/v1/scgrpo/causal/killtrace
declare -A seen
tail -n0 -F $O/signals.log | while read -r line; do
  case "$line" in *"sig=9"*) ;; *) continue ;; esac
  sender=${line%% *}; sender=${sender## }; spid=${sender##*-}; scomm=${sender%-*}
  case "$scomm" in raylet|worker.io|ray::*|claude) continue ;; esac
  [ -n "${seen[$spid]}" ] && continue
  seen[$spid]=1
  {
    echo "===== $(date '+%m-%d %H:%M:%S') sender $sender"
    echo "first: $line"
    ps -o pid,ppid,user,etime,args -p $spid 2>&1 | tail -n +2 | cut -c1-400
    echo "cwd=$(readlink /proc/$spid/cwd 2>/dev/null)"
    echo "cmdline=$(tr '\0' ' ' < /proc/$spid/cmdline 2>/dev/null | cut -c1-600)"
    tr '\0' '\n' < /proc/$spid/environ 2>/dev/null | grep -E '^(USER|SUDO_USER|LOGNAME|SSH_CLIENT|SSH_TTY|PWD|TMUX|STY)='
    p=$(ps -o ppid= -p $spid 2>/dev/null | tr -d ' ')
    while [ -n "$p" ] && [ "$p" -gt 1 ]; do
      echo "parent $p: $(ps -o user=,etime=,args= -p $p 2>/dev/null | cut -c1-300)"
      p=$(ps -o ppid= -p $p 2>/dev/null | tr -d ' ')
    done
  } >> $O/killers.log 2>&1
done
