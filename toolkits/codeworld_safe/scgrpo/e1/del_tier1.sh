#!/bin/bash
# Tier 1 (user-approved 10-30): ModelScope staging copies, all verified on the remote.
# Refuses to run while an upload script still uses a staging directory.
cd /data_cf
if ps -eo args | grep -qE "[u]pload_(e1e8|e8b|causal)_ms"; then echo "upload still running, abort"; exit 3; fi
grep -q "ALL VERIFIED" cws_ms_stage_causal/upload.log || { echo "causal staging not verified, abort"; exit 4; }
grep -q "ALL VERIFIED (26 units)" cws_ms_stage_e1e8/upload.log || { echo "e1e8 staging not verified, abort"; exit 4; }
df -h /data_cf | tail -1
for d in cws_ms_stage_e1e8 cws_ms_stage_causal; do
  [ -d "/data_cf/$d" ] && rm -rf -- "/data_cf/$d" && echo "deleted /data_cf/$d"
done
for f in cws_ms_stage_console.log cws_ms_stage_causal_run.log cws_ms_stage_rmsam_console.log; do
  [ -f "/data_cf/$f" ] && rm -f -- "/data_cf/$f" && echo "deleted /data_cf/$f"
done
df -h /data_cf | tail -1
