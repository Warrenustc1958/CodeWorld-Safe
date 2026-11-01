#!/bin/bash
# Stop the osmesa E12-R run and restart it from step 0 with EGL rendering (GPUs 3/4 render on 2/5).
# Old outputs are moved aside, not deleted.
S=/data_cf/cws_sam/v1/scgrpo
C=/mnt/nfs/fjj/Code/CodeWorld-Safe/examples/embodiment/config/libero_safety_scgrpo_causal_e12r.yaml
cd $S
systemctl stop cws-e12-watch.timer cws-e12.service
for d in /proc/[0-9]*; do  # the launcher (TAG=e12r in its environment)
  tr '\0' ' ' < $d/cmdline 2>/dev/null | grep -q "launch_train.sh" || continue
  tr '\0' '\n' < $d/environ 2>/dev/null | grep -qx "TAG=e12r" && kill -TERM ${d#/proc/} && echo "killed launcher ${d#/proc/}"
done
sleep 3
TAG=e12r bash stop_causal_head.sh
all=""
for d in /proc/[0-9]*; do
  tr '\0' '\n' < $d/environ 2>/dev/null | grep -qx "RAY_TMPDIR=/tmp/rp_ce12r" && all="$all ${d#/proc/}"
done
all=$(echo $all | tr ' ' '\n' | grep -vx "$$" | sort -u)
[ -n "$all" ] && { kill -TERM $all 2>/dev/null; sleep 8; for q in $all; do kill -0 $q 2>/dev/null && kill -KILL $q; done; }
echo "left rp_ce12r processes: $(for d in /proc/[0-9]*; do tr '\0' '\n' < $d/environ 2>/dev/null | grep -qx RAY_TMPDIR=/tmp/rp_ce12r && echo x; done | wc -l)"
T=$(date +%m%d_%H%M)
mkdir -p causal/e12r_osmesa_$T
mv causal/e12r causal/e12r_osmesa_$T/run 2>/dev/null
mv causal/train_e12r*.log causal/e12r_driver.log causal/e12r_osmesa_$T/ 2>/dev/null
mv causal/e12night causal/e12r_osmesa_$T/e12night 2>/dev/null
mkdir -p causal/e12night
# render backend overridable (default osmesa, as before)
sed -i 's#MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa#MUJOCO_GL=${RENDER_GL:-osmesa} PYOPENGL_PLATFORM=${RENDER_GL:-osmesa}#' run_causal.sh
grep -n "MUJOCO_GL" run_causal.sh
grep -q "render_egl_remap" $C || sed -i '/^    cost_label: grader/a\    render_egl_remap: {3: 2, 4: 7}   # CUDA 3 -> EGL 2 (CUDA 2), CUDA 4 -> EGL 7 (CUDA 5); GPUs 3/4 have EGL faults' $C
grep -n "obs_source\|render_egl_remap\|resume_dir" $C
systemctl start cws-e12.service cws-e12-watch.timer
sleep 40
echo "e12 $(systemctl is-active cws-e12.service), timer $(systemctl is-active cws-e12-watch.timer)"
tail -3 causal/e12night/pipeline.log
