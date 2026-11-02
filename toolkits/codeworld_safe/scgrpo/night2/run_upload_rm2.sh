#!/bin/bash
# Background launcher for upload_rm2_ms.py: P1 first (RMs, RM dataset, keyframe scan, key checkpoints), then all.
cd /data_cf/cws_sam/v1/scgrpo
export MS_TOKEN_FILE=/mnt/nfs/fjj/.modelscope_fjj/sdk_token_cws
/usr/bin/python3 upload_rm2_ms.py --only P1 && /usr/bin/python3 upload_rm2_ms.py
echo "UPLOAD_EXIT=$?"
