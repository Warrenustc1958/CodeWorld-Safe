"""Minimal Anthropic-Messages client shared by the three CodeWorld-Safe LLM providers.

Keys are read from the environment variable named in llm_providers.yaml and are
never logged, persisted, or included in exceptions.

Usage:
    from llm_client import LLMClient
    client = LLMClient()                       # provider from CWS_LLM_PROVIDER or yaml default
    text = client.complete(system, user, role="planner")

Self-check (one tiny request, prints provider/model/latency only):
    CWS_TOKENHUB_API_KEY=... python llm_client.py --provider tokenhub --ping
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

_CONFIG = Path(__file__).with_name("llm_providers.yaml")
_ANTHROPIC_VERSION = "2023-06-01"


class LLMError(RuntimeError):
    """Request failure; the message never contains the API key."""


class LLMClient:
    def __init__(self, provider: str | None = None, config_path: Path = _CONFIG) -> None:
        cfg = yaml.safe_load(config_path.read_text())
        name = provider or os.environ.get("CWS_LLM_PROVIDER") or cfg["default_provider"]
        if name not in cfg["providers"]:
            raise LLMError(f"unknown provider {name!r}; choose from {sorted(cfg['providers'])}")
        self.name = name
        self.p = cfg["providers"][name]
        self.req = cfg["request"]
        if self.p.get("api_format") != "anthropic":
            raise LLMError(f"provider {name} uses unsupported api_format {self.p.get('api_format')!r}")
        key = os.environ.get(self.p["key_env"])
        if not key:
            raise LLMError(f"set {self.p['key_env']} in the environment to use provider {name}")
        self._key = key
        if self.p.get("no_proxy"):
            hosts = os.environ.get("NO_PROXY", "")
            os.environ["NO_PROXY"] = ",".join(filter(None, [hosts, *self.p["no_proxy"]]))

    def model(self, role: str) -> str:
        return self.p["models"][role]

    def complete(self, system: str, user: str, role: str = "planner", max_tokens: int | None = None) -> str:
        body = {
            "model": self.model(role),
            "max_tokens": max_tokens or self.req["max_tokens"],
            "temperature": self.req["temperature"],
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        headers = {
            "content-type": "application/json",
            "anthropic-version": _ANTHROPIC_VERSION,
            "x-api-key": self._key,
            "authorization": f"Bearer {self._key}",  # some gateways read this instead of x-api-key
            **self.p.get("extra_headers", {}),
        }
        url = self.p["base_url"].rstrip("/") + "/v1/messages"
        last: Exception | None = None
        for attempt in range(self.req["max_retries"]):
            try:
                req = urllib.request.Request(url, json.dumps(body).encode(), headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.req["timeout_s"]) as r:
                    data = json.load(r)
                return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
            except urllib.error.HTTPError as e:
                detail = e.read()[:300].decode(errors="replace")
                last = LLMError(f"{self.name}: HTTP {e.code} {detail}")
                if e.code < 500 and e.code != 429:
                    break
            except (urllib.error.URLError, TimeoutError) as e:
                last = LLMError(f"{self.name}: {type(e).__name__}: {getattr(e, 'reason', e)}")
            time.sleep(2 ** attempt)
        raise last or LLMError(f"{self.name}: request failed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider")
    ap.add_argument("--ping", action="store_true")
    args = ap.parse_args()
    client = LLMClient(args.provider)
    if args.ping:
        t = time.time()
        out = client.complete("Reply with the single word: pong", "ping", role="worker", max_tokens=8)
        print(json.dumps({"provider": client.name, "model": client.model("worker"),
                          "reply": out.strip()[:20], "latency_s": round(time.time() - t, 2)}))


if __name__ == "__main__":
    main()
