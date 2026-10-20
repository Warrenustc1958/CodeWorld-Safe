"""Re-ingest completed Step E runs after the near-miss channel fix (backups kept)."""
import json
import shutil
import time
from pathlib import Path

from toolkits.codeworld_safe.agent.archive import Archive
from toolkits.codeworld_safe.agent import campaign_step_e as c

root = Path("/data_cf/cws/step_e")
adir = root / "archive"
tag = time.strftime("%Y%m%d_%H%M%S")
for name in ("outcomes.jsonl",):
    if (adir / name).exists():
        shutil.copy2(adir / name, adir / f"{name}.bak_{tag}")
        (adir / name).write_text("")
if (root / "state.json").exists():
    shutil.copy2(root / "state.json", root / f"state.json.bak_{tag}")
state = {}
arch = Archive(adir)
for cell_dir in sorted((root / "runs").iterdir()):
    for run in sorted(p for p in cell_dir.iterdir() if p.is_dir()):
        log = run / "launcher.log"
        if not (log.exists() and "EXIT=0" in log.read_text(errors="ignore")):
            dest = root / "incomplete" / cell_dir.name / run.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(run), str(dest))
            print("moved incomplete", run, "->", dest)
            continue
        stats = c.ingest_run(run, arch, cell_dir.name, run.name)
        cnt = c.counts(arch, cell_dir.name)
        st = state.setdefault(cell_dir.name, {"rounds": 0, "stagnant": 0, "status": "active",
                                              "reason": "", "runs": []})
        st["rounds"] += 1
        st["runs"].append({"run": run.name, "program": None, "rc": 0, "stats": stats, "counts": cnt})
        print(cell_dir.name, run.name, stats, cnt)
(root / "state.json").write_text(json.dumps(state, indent=1))
rep = c.report(root, arch, state)
print("report", {k: rep[k] for k in ("cells_done", "mean_quota_rate", "total_sim_steps", "unsafe_per_10k_steps")})
