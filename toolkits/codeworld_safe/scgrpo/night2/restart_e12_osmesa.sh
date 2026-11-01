#!/bin/bash
# Stop the EGL E12-R run and restart it from step 0 with osmesa rendering, 10 steps, eval step 10 only.
# Old outputs are moved aside, not deleted.
S=/data_cf/cws_sam/v1/scgrpo
C=/mnt/nfs/fjj/Code/CodeWorld-Safe/examples/embodiment/config/libero_safety_scgrpo_causal_e12r.yaml
cd $S
systemctl stop cws-e12-watch.timer cws-e12.service
for d in /proc/[0-9]*; do  # the launcher (TAG=e12r in its environment)
  tr '\0' ' ' < $d/cmdline 2>/dev/null | grep -q "launch_train.sh" || continue
  tr '\0' '\n' < $d/environ 2>/dev/null | grep -qx "TAG=e12r" && kill -TERM ${d#/proc/} && echo "killed launcher ${d#/proc/}"
done 2>/dev/null
sleep 3
TAG=e12r bash stop_causal_head.sh
all=""
for d in /proc/[0-9]*; do
  tr '\0' '\n' < $d/environ 2>/dev/null | grep -qx "RAY_TMPDIR=/tmp/rp_ce12r" && all="$all ${d#/proc/}"
done 2>/dev/null
all=$(echo $all | tr ' ' '\n' | grep -vx "$$" | sort -u)
[ -n "$all" ] && { kill -TERM $all 2>/dev/null; sleep 8; for q in $all; do kill -0 $q 2>/dev/null && kill -KILL $q; done; }
echo "left rp_ce12r processes: $(for d in /proc/[0-9]*; do tr '\0' '\n' < $d/environ 2>/dev/null | grep -qx RAY_TMPDIR=/tmp/rp_ce12r && echo x; done 2>/dev/null | wc -l)"
T=$(date +%m%d_%H%M)
mkdir -p causal/e12r_egl_$T
mv causal/e12r causal/e12r_egl_$T/run 2>/dev/null
mv causal/train_e12r*.log causal/e12r_driver.log causal/e12r_egl_$T/ 2>/dev/null
mv causal/e12night causal/e12r_egl_$T/e12night 2>/dev/null
mkdir -p causal/e12night
sed -i -e "s#^  max_steps: 20#  max_steps: 10#" -e "s#^  resume_dir: .*#  resume_dir: null#" $C
grep -n "max_steps:\|resume_dir\|obs_source" $C | head -4
grep -n "RENDER_GL\|KEEP=\|for s in" e12_pipeline.sh
systemctl start cws-e12.service cws-e12-watch.timer
sleep 40
echo "e12 $(systemctl is-active cws-e12.service), timer $(systemctl is-active cws-e12-watch.timer)"
tail -2 causal/e12night/pipeline.log
for d in /proc/[0-9]*; do
  tr '\0' '\n' < $d/environ 2>/dev/null | grep -qx "RAY_TMPDIR=/tmp/rp_ce12r" || continue
  tr '\0' '\n' < $d/environ | grep -m1 "^MUJOCO_GL=" && break
done 2>/dev/null
