# D2 analysis: affordance RM v3 vs real goal on on-policy rollouts (fixed 64-step window)
import glob, json, pickle, sys
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "/data_cf/cws_sam/v1/scgrpo/diag/d2"
RULES = [(0.2, 2), (0.5, 3), (0.8, 4)]
G = 8


def detect(p, thr, k):
    """first step where k consecutive frames inside one 8-frame chunk have p >= thr; -1 if never"""
    B, T = p.shape
    first = np.full(B, -1)
    for i in range(B):
        for c in range(T // 8):
            run = 0
            for s in range(c * 8, c * 8 + 8):
                run = run + 1 if p[i, s] >= thr else 0
                if run >= k:
                    first[i] = s
                    break
            if first[i] >= 0:
                break
    return first


def gadv(r, gid):
    a = np.zeros_like(r, dtype=float)
    for g in np.unique(gid):
        m = gid == g
        a[m] = (r[m] - r[m].mean()) / (r[m].std() + 1e-6)
    return a


out = {}
for pdir in sorted(glob.glob(f"{D}/*/")):
    name = pdir.rstrip("/").split("/")[-1]
    fs = sorted(glob.glob(pdir + "batch_*.pkl"))
    if not fs:
        continue
    P, SR, AR, GID = [], [], [], []
    for j, f in enumerate(fs):
        r = pickle.load(open(f, "rb"))
        P.append(r["rm_v3_p"]); SR.append(r["step_rec"]); AR.append(r["at_reset_success"])
        GID.append(r["origin_index"] + 1000 * j)
    p, sr, ar, gid = map(np.concatenate, (P, SR, AR, GID))
    keep = ~ar.astype(bool)
    p, sr, gid = p[keep], sr[keep], gid[keep]
    goal = sr[:, :, 0].astype(bool); wrong = sr[:, :, 1].astype(bool); viol = sr[:, :, 2].astype(bool)
    g_any = goal.any(1)
    g_first = np.where(g_any, goal.argmax(1), 99)
    res = {"n_batches": len(fs), "n_eps": int(len(p)), "real_goal_rate": round(g_any.mean(), 3),
           "wrongpart_rate": round(wrong.any(1).mean(), 3), "violation_rate": round(viol.any(1).mean(), 3),
           "goal_and_clean_rate": round((g_any & ~wrong.any(1)).mean(), 3)}
    for thr, k in RULES:
        fd = detect(p, thr, k); d = fd >= 0
        tp = (d & g_any).sum(); fp = (d & ~g_any).sum(); fn = (~d & g_any).sum()
        fp_wrong = (d & ~g_any & wrong.any(1)).sum()
        fp_nogoal = fp - fp_wrong
        early = (d & g_any & (fd < g_first - 7)).sum()        # fires >1 chunk before the real goal
        late = (d & g_any & (fd > g_first + 15)).sum()         # fires >2 chunks after
        # within-group advantage-sign agreement (RM binary reward vs real goal reward)
        rr, rm = g_any.astype(float), d.astype(float)
        ar_, am_ = gadv(rr, gid), gadv(rm, gid)
        # sign agreement only where both rewards give a nonzero advantage
        nz = (np.abs(ar_) > 1e-3) & (np.abs(am_) > 1e-3)
        agree = (np.sign(ar_) == np.sign(am_))[nz].mean() if nz.any() else float("nan")
        opposite = ((np.sign(ar_) * np.sign(am_)) < 0)[nz].mean() if nz.any() else float("nan")
        groups = np.unique(gid)
        res_both = np.mean([(rm[gid == g].std() > 0) and (rr[gid == g].std() > 0) for g in groups])
        spurious = np.mean([(rm[gid == g].std() > 0) and (rr[gid == g].std() == 0) for g in groups])
        missed = np.mean([(rm[gid == g].std() == 0) and (rr[gid == g].std() > 0) for g in groups])
        corr = float(np.corrcoef(ar_, am_)[0, 1]) if ar_.std() > 0 and am_.std() > 0 else float("nan")
        res[f"{thr}x{k}"] = {
            "precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3),
            "rm_fire_rate": round(d.mean(), 3), "FP_wrong_part": int(fp_wrong), "FP_no_goal": int(fp_nogoal),
            "TP_premature": int(early), "TP_late": int(late), "TP": int(tp), "FN": int(fn),
            "adv_sign_agree": round(float(agree), 3), "adv_sign_opposite": round(float(opposite), 3),
            "adv_corr": round(corr, 3), "groups_both_signal": round(float(res_both), 3),
            "groups_real_signal": round(float(np.mean([rr[gid == g].std() > 0 for g in groups])), 3),
            "groups_spurious_signal": round(float(spurious), 3),
            "groups_missed_signal": round(float(missed), 3)}
    out[name] = res
    print(name, json.dumps(res, indent=1))

json.dump(out, open(f"{D}/../d2_summary.json", "w"), indent=1)
