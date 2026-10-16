"""Compare two LIBERO-Safety paper-eval roots cell by cell (last value in metrics.log).

Usage: python cmp_wrist_eval.py <baseline_root> <ablation_root>
"""
import re
import sys
from pathlib import Path

KEYS = ["success_once", "safe_success_once", "safety_violation_once", "collision_once"]
SHORT = {"success_once": "SR", "safe_success_once": "SSR",
         "safety_violation_once": "Viol", "collision_once": "Coll"}


def read(cell_dir: Path):
    m = cell_dir / "metrics.log"
    if not m.is_file():
        return None
    text = m.read_text(errors="ignore")
    out = {}
    for k in KEYS:
        vals = re.findall(rf"(?<![\w]){k}=([-\d.eE]+)", text)
        out[k] = float(vals[-1]) if vals else None
    return out


def main():
    base, abl = Path(sys.argv[1]), Path(sys.argv[2])
    cells = sorted(d.name for d in base.iterdir() if (d / "metrics.log").is_file())
    hdr = " | ".join(f"{SHORT[k]} dual/nowrist(d)" for k in KEYS)
    print(f"cell | {hdr}")
    sums = {k: [0.0, 0.0, 0] for k in KEYS}
    for c in cells:
        b, a = read(base / c), read(abl / c)
        cols = []
        for k in KEYS:
            bv = b.get(k) if b else None
            av = a.get(k) if a else None
            if bv is None or av is None:
                cols.append(f"{bv}/{av}")
                continue
            cols.append(f"{bv:.2f}/{av:.2f}({av - bv:+.2f})")
            sums[k][0] += bv
            sums[k][1] += av
            sums[k][2] += 1
        print(f"{c} | " + " | ".join(cols))
    cols = []
    for k in KEYS:
        sb, sa, n = sums[k]
        cols.append(f"{sb / n:.3f}/{sa / n:.3f}({(sa - sb) / n:+.3f})" if n else "-")
    print("MEAN | " + " | ".join(cols))


if __name__ == "__main__":
    main()
