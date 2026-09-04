#!/usr/bin/env python3
"""
TIGER SQ-AI 2026 — container-vs-PUBLISHED-contract self-test verifier (2026-07-19).

Checks a produced /output tree against the published Docker-Instructions contract
(wiki 639935 / metrics/00_README.md @origin/main):
  * task1/ = MERGED  (15/16-cls) exact-RGB PNG   [official "Simplified"]
  * task2/ = FINE    (30/31-cls) exact-RGB PNG   [official "Full"]
  * task3.csv         14 stations 6L..13R, continuous [0,1], case_id = stem

Verifies (per the contract's "common reasons a submission fails" list):
  1. task1 round-trips 0-unknown under the MERGED registry; task2 0-unknown under FINE.
  2. output PNG size == input frame size (native + novel/variable resolutions).
  3. task3.csv header exact, one row/frame, all scores in [0,1].
  4. a sanity weighted-Dice (160px, OFFICIAL both-empty/one-empty semantics, per-case
     mean) vs the correct GT to confirm the routing is scored correctly (not collapsed).
  --make-downscaled builds 256px gt/ + pred/<M>/ trees for the OFFICIAL 01_evaluate script.

Exit code 0 iff every hard check PASSes.
"""
import argparse, csv, glob, os, sys, json
import numpy as np
from PIL import Image

def load_registries(repo):
    sys.path.insert(0, repo)
    from metrics.classes import CLASSES as FINE, rgb_mask_to_label_mask as rgb2fine
    from metrics.classes_merged import CLASSES_MERGED as MERG, rgb_mask_to_label_mask as rgb2merg
    return FINE, rgb2fine, MERG, rgb2merg

def ds(lab, m=160):
    h, w = lab.shape
    if max(h, w) <= m: return lab
    s = m / max(h, w)
    return np.array(Image.fromarray(lab.astype(np.uint8)).resize(
        (max(1, int(w*s)), max(1, int(h*s))), Image.NEAREST))

def unknown(rgb_dir, rgb2lab):
    unk = tot = 0
    for p in sorted(glob.glob(rgb_dir+"/*.png")):
        lab = rgb2lab(np.array(Image.open(p).convert("RGB")))
        unk += int((lab == 255).sum()); tot += lab.size
    return unk, tot

def size_ok(out_dir, in_dir):
    bad = []
    for p in sorted(glob.glob(out_dir+"/*.png")):
        n = os.path.basename(p)
        ip = os.path.join(in_dir, n)
        if not os.path.exists(ip): continue
        if Image.open(p).size != Image.open(ip).size:
            bad.append((n, Image.open(p).size, Image.open(ip).size))
    return bad

def case_of(stem):  # everything before the last _
    return stem.rsplit("_", 1)[0]

def weighted_dice(pred_dir, gt_dir, classes, rgb2lab, dsz=160):
    """OFFICIAL semantics: both-empty=1.0, one-empty=0.0; per-image weighted macro
    (Sum w_c*dice_c / Sum w_c over ALL classes) + separate w3 sub-avg; then case-mean."""
    wt = {c.label_id: c.weight for c in classes}
    w3 = [c.label_id for c in classes if c.weight == 3]
    per_case_w, per_case_w3 = {}, {}
    for p in sorted(glob.glob(pred_dir+"/*.png")):
        n = os.path.basename(p); gp = os.path.join(gt_dir, n)
        if not os.path.exists(gp): continue
        pr = ds(rgb2lab(np.array(Image.open(p).convert("RGB"))), dsz)
        gt = ds(rgb2lab(np.array(Image.open(gp).convert("RGB"))), dsz)
        dice = {}
        for c in classes:
            cid = c.label_id
            g = (gt == cid); q = (pr == cid)
            gs, qs = int(g.sum()), int(q.sum())
            if gs == 0 and qs == 0: dice[cid] = 1.0
            elif gs == 0 or qs == 0: dice[cid] = 0.0
            else: dice[cid] = 2*int((g & q).sum())/(gs+qs)
        num = sum(wt[c]*dice[c] for c in dice); den = sum(wt.values())
        img_w = num/den
        img_w3 = float(np.mean([dice[c] for c in w3])) if w3 else float("nan")
        cs = case_of(os.path.splitext(n)[0])
        per_case_w.setdefault(cs, []).append(img_w)
        per_case_w3.setdefault(cs, []).append(img_w3)
    overall = float(np.mean([np.mean(v) for v in per_case_w.values()]))
    w3avg = float(np.mean([np.mean(v) for v in per_case_w3.values()]))
    return overall, w3avg

def make_downscaled(out_root, native_out, gt_root, method, dsz=256):
    """Build 256px gt/ + pred/<method>/ trees for the OFFICIAL 01_evaluate_challenge.py."""
    gt_map = {"task1": gt_root+"/masks_coarse", "task2": gt_root+"/masks_fine"}
    for sub in ("gt", f"pred/{method}"):
        for t in ("task1", "task2"):
            os.makedirs(f"{out_root}/{sub}/{t}", exist_ok=True)
    def down_rgb(src, dst):
        im = Image.open(src).convert("RGB"); w, h = im.size
        s = dsz/max(w, h); nw, nh = max(1, int(w*s)), max(1, int(h*s))
        im.resize((nw, nh), Image.NEAREST).save(dst)
    for t in ("task1", "task2"):
        for p in sorted(glob.glob(f"{native_out}/{t}/*.png")):
            n = os.path.basename(p)
            down_rgb(p, f"{out_root}/pred/{method}/{t}/{n}")
            gtp = os.path.join(gt_map[t], n)
            if os.path.exists(gtp): down_rgb(gtp, f"{out_root}/gt/{t}/{n}")
    # task3 gt + pred passthrough
    import shutil
    shutil.copy(f"{native_out}/task3.csv", f"{out_root}/pred/{method}/task3.csv")
    # build gt task3.csv from visibility csv (stems present in native_out)
    stems = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(f"{native_out}/task1/*.png")]
    build_gt_task3(gt_root, stems, f"{out_root}/gt/task3.csv")

def build_gt_task3(gt_root, stems, out_csv):
    """Parse the OFFICIAL visibility CSV: columns = case, annotated-frame(station),
    'visible lymph stations' (comma-sep list). GT stem = f'{case}_{frame_station}',
    binary label[st]=1 if st in the visible list."""
    STAT = ["6L","6R","7L","7R","8","9","10L","10R","11L","11R","12L","12R","13L","13R"]
    vis = gt_root+"/lymph_node_station_visibility.csv"
    lut = {}
    if os.path.exists(vis):
        with open(vis) as f:
            r = csv.reader(f); hdr = next(r)
            for row in r:
                if len(row) < 3: continue
                case, frame, vlist = row[0].strip(), row[1].strip(), row[2]
                stem = f"{case}_{frame}"
                visible = {s.strip() for s in vlist.replace('"', '').split(",") if s.strip()}
                lut[stem] = visible
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["case_id"]+STAT)
        for s in stems:
            visible = lut.get(s, set())
            w.writerow([s]+[1 if st in visible else 0 for st in STAT])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--native-out", required=True)
    ap.add_argument("--native-in", required=True)
    ap.add_argument("--varres-out", default=None)
    ap.add_argument("--varres-in", default=None)
    ap.add_argument("--make-downscaled", default=None, help="root for 256px official-eval trees")
    ap.add_argument("--verdict", default=None)
    a = ap.parse_args()
    FINE, rgb2fine, MERG, rgb2merg = load_registries(a.repo)
    STAT = ["6L","6R","7L","7R","8","9","10L","10R","11L","11R","12L","12R","13L","13R"]
    checks = {}

    def run_tree(tag, out, inp):
        u1, t1 = unknown(out+"/task1", rgb2merg)   # task1 = MERGED
        u2, t2 = unknown(out+"/task2", rgb2fine)   # task2 = FINE
        checks[f"{tag}_task1_merged_0unknown"] = (u1 == 0, f"{u1}/{t1} unknown")
        checks[f"{tag}_task2_fine_0unknown"]   = (u2 == 0, f"{u2}/{t2} unknown")
        b1 = size_ok(out+"/task1", inp); b2 = size_ok(out+"/task2", inp)
        checks[f"{tag}_task1_size_match"] = (len(b1) == 0, f"{len(b1)} mismatched" + (str(b1[:2]) if b1 else ""))
        checks[f"{tag}_task2_size_match"] = (len(b2) == 0, f"{len(b2)} mismatched" + (str(b2[:2]) if b2 else ""))

    run_tree("native", a.native_out, a.native_in)
    if a.varres_out and a.varres_in:
        run_tree("varres", a.varres_out, a.varres_in)

    # task3 schema (native)
    with open(a.native_out+"/task3.csv") as f:
        r = csv.reader(f); hdr = next(r); rows = list(r)
    hdr_ok = hdr == ["case_id"]+STAT
    rng_ok = all(len(row) == 15 and all(0.0 <= float(v) <= 1.0 for v in row[1:]) for row in rows)
    n_native = len(glob.glob(a.native_out+"/task1/*.png"))
    checks["task3_header_exact"] = (hdr_ok, str(hdr))
    checks["task3_values_in_0_1"] = (rng_ok, "")
    checks["task3_one_row_per_frame"] = (len(rows) == n_native, f"{len(rows)} rows / {n_native} frames")

    # sanity weighted-Dice (native, 160px, OFFICIAL semantics)
    m_overall, m_w3 = weighted_dice(a.native_out+"/task1", a.gt_root+"/masks_coarse", MERG, rgb2merg)
    f_overall, f_w3 = weighted_dice(a.native_out+"/task2", a.gt_root+"/masks_fine", FINE, rgb2fine)
    print(f"[SANITY DICE 160px, official both-empty semantics, case-mean]")
    print(f"  task1 MERGED: overall wDice={m_overall:.4f}  w3 sub-avg={m_w3:.4f}   (dossier ~0.518 / w3 0.552)")
    print(f"  task2 FINE:   overall wDice={f_overall:.4f}  w3 sub-avg={f_w3:.4f}   (dossier ~0.595 / w3 0.7092)")
    # sane = clearly non-collapsed
    checks["task1_dice_sane_notcollapsed"] = (m_overall > 0.35, f"overall {m_overall:.3f}")
    checks["task2_dice_sane_notcollapsed"] = (f_overall > 0.40, f"overall {f_overall:.3f}")

    if a.make_downscaled:
        make_downscaled(a.make_downscaled, a.native_out, a.gt_root, "AIMS", 256)
        print(f"[downscaled 256px official-eval trees written to {a.make_downscaled}]")

    print("\n=== SELF-TEST CHECKS ===")
    allpass = True
    for k, (ok, detail) in checks.items():
        allpass &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}  {detail}")
    print(f"\n=== OVERALL: {'PASS' if allpass else 'FAIL'} ===")
    if a.verdict:
        json.dump({"pass": allpass, "checks": {k: v[0] for k, v in checks.items()},
                   "dice": {"merged_overall": m_overall, "merged_w3": m_w3,
                            "fine_overall": f_overall, "fine_w3": f_w3}},
                  open(a.verdict, "w"), indent=2)
    sys.exit(0 if allpass else 1)

if __name__ == "__main__":
    main()
