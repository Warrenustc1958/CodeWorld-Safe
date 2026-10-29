# D3: is the train/eval logprob mismatch (vision train-mode) large relative to a normal PPO update?
import glob, json, sys, torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as E

G = "/data_cf/cws_sam/v1/scgrpo/diagnostics/rootcause_20261027_v1/gate_isolation/recomputed_logprobs.pt"
d = torch.load(G, map_location="cpu")
q = torch.tensor([0.5, 0.9, 0.99, 1.0])
out = {}


def stats(a, b, name):
    diff = (a - b)  # [64 chunks, 8 steps, 7 dims]
    elem = diff.abs().flatten()
    chunk = diff.sum(dim=(1, 2))  # chunk-level log ratio
    ratio = chunk.exp()
    r = {
        "elem_absdiff_q50/90/99/max": [round(x, 5) for x in torch.quantile(elem, q).tolist()],
        "chunk_logratio_absq50/90/99/max": [round(x, 5) for x in torch.quantile(chunk.abs(), q).tolist()],
        "chunk_ratio_min/max": [round(ratio.min().item(), 4), round(ratio.max().item(), 4)],
        "frac_outside_0.8_1.2": round(((ratio < 0.8) | (ratio > 1.2)).float().mean().item(), 4),
        "frac_outside_0.9_1.1": round(((ratio < 0.9) | (ratio > 1.1)).float().mean().item(), 4),
        "pseudo_kl(mean 0.5*lr^2)": round((0.5 * chunk ** 2).mean().item(), 6),
    }
    out[name] = r
    print(name, json.dumps(r))


stats(d["raw_original16"], d["raw_eval16"], "orig_train_vs_eval@16")
stats(d["raw_fixed16"], d["raw_eval16"], "fixed_vs_eval@16")
stats(d["raw_eval32"], d["raw_eval16"], "eval32_vs_eval16")
stats(d["fsdp_eval16"], d["raw_eval16"], "fsdp_vs_raw@16")
stats(d["raw_nofast_train16"], d["raw_nofast_eval16"], "nofast_train_vs_eval")

# normal update magnitude from training logs
TAGS = ["train/actor/approx_kl", "train/actor/clip_fraction", "train/actor/ratio_abs",
        "train/actor/clipped_ratio", "train/actor/grad_norm", "train/actor/policy_loss_abs"]
runs = sys.argv[1:]
for run in runs:
    fs = sorted(glob.glob(f"{run}/tensorboard/events*"))
    vals = {t: {} for t in TAGS}
    for f in fs:
        e = E(f, size_guidance={"scalars": 0}); e.Reload()
        have = e.Tags()["scalars"]
        for t in TAGS:
            if t in have:
                for s in e.Scalars(t):
                    vals[t][s.step] = s.value
    r = {}
    for t, v in vals.items():
        if not v:
            continue
        x = torch.tensor(list(v.values()))
        r[t.split("/")[-1]] = {"n": len(x), "mean": round(x.mean().item(), 5),
                               "med": round(x.median().item(), 5), "max": round(x.max().item(), 5)}
    out[run] = r
    print(run.split("/")[-1], json.dumps(r))

json.dump(out, open("/data_cf/cws_sam/v1/scgrpo/diag/d3_summary.json", "w"), indent=1)
