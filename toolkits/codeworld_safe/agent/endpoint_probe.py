#!/usr/bin/env python3
"""Raw-response probe for LLM endpoints (debugging only). Keys come from the environment
and are never printed. Prints HTTP status, top-level keys and a truncated body.

  CWS_TOKENHUB_API_KEY=... python3 endpoint_probe.py tokenhub
  CWS_AZURE_API_KEY=...    python3 endpoint_probe.py azure
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

PROMPT = "Reply with the single word: pong"
TOKENHUB = "https://thc.alibaba-inc.com/compatible/v1/messages"
AZURE = "https://fujiajun-steplatent-gpt.openai.azure.com/openai"


def tokenhub_probes(key: str):
    h = {"content-type": "application/json", "anthropic-version": "2023-06-01",
         "x-api-key": key, "authorization": f"Bearer {key}"}
    for model in ("model_8t9h8j", "model_c3udx4"):
        base = {"model": model, "messages": [{"role": "user", "content": PROMPT}]}
        # Vary the knobs that most often trip compat gateways: max_tokens floor, temperature, system.
        yield f"{model} max64", TOKENHUB, h, {**base, "max_tokens": 64}
        yield f"{model} max1024+temp+system", TOKENHUB, h, {**base, "max_tokens": 1024, "temperature": 0.4,
                                                            "system": "You are terse."}


def azure_probes(key: str):
    h = {"content-type": "application/json", "api-key": key}
    yield ("responses", f"{AZURE}/responses?api-version=2025-04-01-preview", h,
           {"model": "gpt-4o", "input": PROMPT, "max_output_tokens": 16})
    yield ("chat/gpt-4o", f"{AZURE}/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview", h,
           {"messages": [{"role": "user", "content": PROMPT}], "max_tokens": 8})


def run(name, url, headers, body, key):
    t = time.time()
    req = urllib.request.Request(url, json.dumps(body).encode(), headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=240) as r:
            status, raw = r.status, r.read()
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read()
    except Exception as e:  # network-level failure
        print(json.dumps({"probe": name, "error": f"{type(e).__name__}: {e}"}))
        return
    text = raw.decode(errors="replace").replace(key, "***")
    try:
        keys = sorted(json.loads(text))
    except ValueError:
        keys = "non-json"
    print(json.dumps({"probe": name, "http": status, "latency_s": round(time.time() - t, 1),
                      "top_keys": keys, "body": text[:700]}, ensure_ascii=False))


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else ""
    env = {"tokenhub": "CWS_TOKENHUB_API_KEY", "azure": "CWS_AZURE_API_KEY"}.get(target)
    if not env or not os.environ.get(env):
        sys.exit("usage: <KEY_ENV>=... endpoint_probe.py tokenhub|azure")
    key = os.environ[env]
    probes = tokenhub_probes(key) if target == "tokenhub" else azure_probes(key)
    for name, url, headers, body in probes:
        run(name, url, headers, body, key)


if __name__ == "__main__":
    main()
