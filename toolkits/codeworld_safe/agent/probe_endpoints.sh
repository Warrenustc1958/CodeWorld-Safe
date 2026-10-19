#!/usr/bin/env bash
# Interactive endpoint probe. Keys are read with `read -s`; press Enter to skip one.
#   bash /mnt/nfs/fjj/Code/CodeWorld-Safe/toolkits/codeworld_safe/agent/probe_endpoints.sh
set -u
cd "$(dirname "$0")"
OUT=/mnt/nfs/fjj/Code/rlinf-safety-runtime/tmp/cws_a0
mkdir -p "$OUT"
LOG="$OUT/probe_$(date +%Y%m%d_%H%M%S).log"
read -r -s -p "CWS_TOKENHUB_API_KEY (Enter to skip): " TH; echo
read -r -s -p "CWS_AZURE_API_KEY (Enter to skip): " AZ; echo
{
  if [ -n "$TH" ]; then echo "== tokenhub"; CWS_TOKENHUB_API_KEY="$TH" python3 endpoint_probe.py tokenhub; fi
  if [ -n "$AZ" ]; then echo "== azure"; CWS_AZURE_API_KEY="$AZ" python3 endpoint_probe.py azure; fi
} 2>&1 | tee "$LOG"
unset TH AZ
echo "log: $LOG"
