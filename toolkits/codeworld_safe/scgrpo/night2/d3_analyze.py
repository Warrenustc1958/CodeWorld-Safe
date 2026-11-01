"""D3: within-group cost signal, violation timing and SAM alignment from per-episode dumps (sync_real_env dump_dir).

  python d3_analyze.py DUMP_DIR [BUDGET]   -> report on stdout
Groups = (worker file, reset, group). Channels c_obj / c_robot / c_grasp use the cost window (start .. first goal).
"""
import glob
import json
import os
import sys
from collections import defaultdict

CH = ("obj", "robot", "grasp")


def auroc(pos, neg):
    """Mann-Whitney AUROC (ties count half); None if a class is empty."""
    if not pos or not neg:
        return None
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank_sum, i = 0.0, 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        r = (i + j + 1) / 2.0  # average 1-based rank of the tie block
        rank_sum += r * sum(1 for k in range(i, j) if allv[k][1])
        i = j
    n1, n0 = len(pos), len(neg)
    return (rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def main(d, budget):
    rows, groups = [], defaultdict(list)
    for f in sorted(glob.glob(os.path.join(d, "episodes_*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            rows.append(r)
            groups[(os.path.basename(f), r["reset"], r["group"])].append(r)
    n = len(rows)
    print(f"{n} episodes, {len(groups)} groups (sizes {sorted({len(g) for g in groups.values()})})")
    if not n:
        return

    print("\n## Within-group signal (cost window)")
    print("| signal | rate | groups all 0 | groups all 1 | mixed groups |")
    print("|---|---|---|---|---|")
    sigs = {"real_success": lambda r: int(r["real_success"])}
    for j, c in enumerate(CH):
        sigs[f"c_{c}"] = lambda r, j=j: r["grader_window"][j]
    sigs["any_cost"] = lambda r: int(any(r["grader_window"]))
    for name, fn in sigs.items():
        z = o = m = 0
        for g in groups.values():
            vals = {fn(r) for r in g}
            z += vals == {0}
            o += vals == {1}
            m += len(vals) > 1
        rate = sum(fn(r) for r in rows) / n
        G = len(groups)
        print(f"| {name} | {rate:.3f} | {z / G:.2f} | {o / G:.2f} | {m / G:.2f} |")

    print("\n## Per suite: any-cost rate / mixed-group fraction / robot rate / robot mixed")
    by_suite = defaultdict(list)
    for k, g in groups.items():
        by_suite[g[0].get("suite")].append(g)
    for s, gs in sorted(by_suite.items(), key=lambda x: str(x[0])):
        eps = [r for g in gs for r in g]
        anyc = sum(any(r["grader_window"]) for r in eps) / len(eps)
        mix = sum(len({int(any(r["grader_window"])) for r in g}) > 1 for g in gs) / len(gs)
        rob = sum(r["grader_window"][1] for r in eps) / len(eps)
        rmix = sum(len({r["grader_window"][1] for r in g}) > 1 for g in gs) / len(gs)
        print(f"- {s}: {len(gs)} groups, any cost {anyc:.3f}, mixed {mix:.2f}; robot {rob:.3f}, robot mixed {rmix:.2f}")

    print("\n## Violation timing (first violating step in the 64-step episode)")
    for j, c in enumerate(CH):
        steps = [r["first_viol_step"][j] for r in rows if r["first_viol_step"][j] >= 0]
        at_reset = sum(r["grader_at_reset"][j] for r in rows)
        if steps:
            steps.sort()
            q = lambda p: steps[min(len(steps) - 1, int(p * len(steps)))]
            print(f"- c_{c}: {len(steps)} episodes, at reset {at_reset}, step quartiles {q(.25)}/{q(.5)}/{q(.75)}, "
                  f">=56: {sum(s >= 56 for s in steps)}")
        else:
            print(f"- c_{c}: none (at reset {at_reset})")
    after_goal = sum(1 for r in rows for j in range(3)
                     if r["grader_episode"][j] and not r["grader_window"][j])
    print(f"- channel violations only after the first goal (outside the cost window): {after_goal}")

    print("\n## SAM cost vs grader labels")
    sam = [r["cost_sam"] for r in rows]
    for name, lab in [("any (episode)", lambda r: any(r["grader_episode"])),
                      ("any (window)", lambda r: any(r["grader_window"]))] + \
                     [(f"c_{c} (episode)", lambda r, j=j: r["grader_episode"][j]) for j, c in enumerate(CH)]:
        pos = [r["cost_sam"] for r in rows if lab(r)]
        neg = [r["cost_sam"] for r in rows if not lab(r)]
        a = auroc(pos, neg)
        print(f"- {name}: AUROC {a if a is None else round(a, 3)} (pos {len(pos)}, neg {len(neg)}), "
              f"mean SAM pos {sum(pos) / max(len(pos), 1):.3f} / neg {sum(neg) / max(len(neg), 1):.3f}")
    clean = sorted(r["cost_sam"] for r in rows if not any(r["grader_episode"]))
    true_rate = sum(any(r["grader_window"]) for r in rows) / n
    if clean:
        print(f"- false-positive floor: SAM on violation-free episodes mean {sum(clean) / len(clean):.3f}, "
              f"median {clean[len(clean) // 2]:.3f}, share above budget {budget}: "
              f"{sum(x > budget for x in clean) / len(clean):.2f}")
    print(f"- calibration: mean SAM {sum(sam) / n:.3f} vs true window violation rate {true_rate:.3f}")
    # within-group ranking: does SAM order violating vs clean episodes inside the same group?
    agree = tot = 0
    for g in groups.values():
        v = [r for r in g if any(r["grader_window"])]
        c = [r for r in g if not any(r["grader_window"])]
        for a in v:
            for b in c:
                tot += 1
                agree += (a["cost_sam"] > b["cost_sam"]) + 0.5 * (a["cost_sam"] == b["cost_sam"])
    print(f"- within-group pairwise SAM ranking accuracy: {agree / tot:.3f} over {tot} pairs" if tot
          else "- within-group pairwise ranking: no mixed groups")

    print("\n## RM vs real success (paid reward uses the real outcome here)")
    pos = [max(x for x in r["rm_chunk_max"] if x is not None) for r in rows if r["real_success"]]
    neg = [max(x for x in r["rm_chunk_max"] if x is not None) for r in rows if not r["real_success"]]
    a = auroc(pos, neg)
    fp = sum(r["rm_success_once"] and not r["real_success"] for r in rows)
    print(f"- RM max-prob AUROC {a if a is None else round(a, 3)}; RM false positives {fp}/{len(neg)} failures")
    fp_allfail = 0
    allfail = [g for g in groups.values() if not any(r["real_success"] for r in g)]
    for g in allfail:
        fp_allfail += any(r["rm_success_once"] for r in g)
    print(f"- all-fail groups: {len(allfail)}; with at least one RM false positive: {fp_allfail}")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.101)
