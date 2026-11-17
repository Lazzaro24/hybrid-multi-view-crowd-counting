"""Package the final weights of the thesis for download (LRZ Sync+Share), and check a downloaded copy.

    uv run python scripts/package_final_weights.py            # -> dist/hmvcc_final_weights.zip (+ .sha256)
    uv run python scripts/package_final_weights.py --verify   # check the unpacked files against the checksums

The zip holds the weights at their paths relative to the repository root, so it unpacks there and every command in the README
finds them. It also holds runs/final_weights/MANIFEST.md (which file is which model) and
runs/final_weights/SHA256SUMS.txt; both are also tracked in git, so a download can be checked against the repository.
"""

import argparse
import hashlib
import os
import sys
import zipfile

src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META_DIR = "runs/final_weights"
MANIFEST = f"{META_DIR}/MANIFEST.md"
SUMS = f"{META_DIR}/SHA256SUMS.txt"

# (path relative to the repository root, model, what it is used for)
FILES = [
    (
        "runs/baseline/ot/best.pth",
        "Baseline (OT)",
        "SynMVCrowd baseline after the OT fine-tune; paper protocol, final evaluations",
    ),
    (
        "runs/baseline/run_1/best.pth",
        "Baseline (MSE)",
        "SynMVCrowd baseline trained with MSE (run_1); paper protocol, final evaluations",
    ),
    (
        "runs/full_length_dense_mask_homography/4_router_anchor/"
        "multiscene_final_stage_4_bud200_homography_256_dout0.7_cw0.01_genloss_a0.7_ro_anchor0.1_fs.pth",
        "h1_hom",
        "router lineage on homography_256, Stage 4 final weights; learned routing and oracle routing",
    ),
    (
        "runs/full_length_dense_mask_final/4_router_anchor/"
        "multiscene_final_stage_4_bud200_ca_sa_dout0.7_cw0.01_genloss_a0.7_ro_anchor0.1_fs.pth",
        "h1_casa",
        "router lineage on the ca_sa attention neck, Stage 4 final weights; learned routing and oracle routing",
    ),
    (
        "runs/full_length_hybrid/3_sparse_s42/multiscene_final_stage_2_bud200_homography_256_S4D2_s42.pth",
        "hybrid_hom",
        "dense-first hybrid (D2 dense + S4D2 sparse expert, seed 42); density rule dzone0.1p",
    ),
    (
        "full_length_joint_homography/1_joint/multiscene_final_stage_joint_hom_s0.01_wr1_wf1.pth",
        "J (joint run)",
        "h1_hom trained in one loop without the curriculum; joint-training comparison",
    ),
    (
        "runs/full_length_hybrid/4_h0_control/homography_256/last.pth",
        "H0-control",
        "fusion + baseline decoder fine-tuned for the hybrid's budget; occlusion and map-quality reference",
    ),
    (
        "runs/train_2d/new_best/resnet18_2d_pretrained_ep4.pth",
        "2D backbone",
        "ResNet-18 pretrained on per-camera head density maps; initialises every training, needed to load H0-control "
        "and the H0 short-budget runs",
    ),
    # baselines under the final-weights rule (doc/results.md, "Baselines under the final-weights rule")
    (
        "runs/baseline/ot_frozen/last.pth",
        "Baseline (OT), final weights",
        "last epoch of the OT fine-tune (epoch 65, the early stop); final-weights comparison",
    ),
    (
        "runs/baseline/run_1/last.pth",
        "Baseline (MSE), final weights",
        "last epoch of run_1 (epoch 200); final-weights comparison",
    ),
    # short-budget head ablation on homography_256, same Stage 1 (doc/head_ablation.md sections 8.22-8.24)
    (
        "runs/neck_ablation/ablation_v2/homography_256/last.pth",
        "H0 (short budget)",
        "the 30-epoch homography_256 fusion run with the baseline decoder; Stage 1 of H1-H3, H0 of the head ablation",
    ),
    (
        "runs/head_ablation/homography_256/h0_long/1_decoder/homography_256/last.pth",
        "H0-long (short budget)",
        "H0 trained on for 88 epochs (H1's budget), fusion trainable; budget-matched decoder reference",
    ),
    (
        "runs/head_ablation/homography_256/og_mask/4_router_anchor/"
        "multiscene_final_stage_4_abl_homography_256_og_dout0.7_cw0.01_genloss_a0.7_ro_anchor0.1_fs.pth",
        "H1 (short budget)",
        "learned router, pointwise oracle mask, full recipe incl. Stage 4 anchor; learned routing",
    ),
    (
        "runs/head_ablation/homography_256/window_mask_D0.4/4_router_anchor/"
        "multiscene_final_stage_4_abl_homography_256_W25D0.4_ma50_dout0.7_cw0.01_genloss_a0.7_ro_anchor0.1_fs.pth",
        "H2 (short budget)",
        "learned router, window-density mask W25 / 0.4 p/m^2 / min area 50 px, H1's recipe; learned routing",
    ),
    (
        "runs/head_ablation/homography_256/perf_router/c10_sel/"
        "multiscene_final_stage_3_abl_homography_256_unmasked_joint_genloss_ep30_perf_c10_sel.pth",
        "H3 (short budget)",
        "performance routing, 10 px cells, Stage 3 final weights; learned routing",
    ),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def write_meta(rows):
    os.makedirs(os.path.join(src_dir, META_DIR), exist_ok=True)
    with open(os.path.join(src_dir, SUMS), "w", encoding="utf-8", newline="\n") as f:
        for rel, digest, _ in rows:
            f.write(f"{digest}  {rel}\n")  # sha256sum format, paths relative to the repository root
    lines = [
        "# Final weights of the thesis",
        "",
        "Download: LRZ Sync+Share, link in the README (section 4, *Final weights*). Unzip into the repository root; the files land",
        "at the paths below, which are the paths the README's commands use. Check them with",
        "`uv run python scripts/package_final_weights.py --verify` (from the repository root) or `sha256sum -c " + SUMS + "`.",
        "",
        "The seven final evaluations use the first five files; the two oracle runs reuse the h1_hom and h1_casa",
        "weights (oracle routing on the Stage 4 weights reproduces the Stage 3 oracle exactly, see",
        "`doc/final_evaluations.md` section 7.4). The two baseline *final weights* rows are the last epochs of the same",
        "baseline runs (`doc/results.md`, *Baselines under the final-weights rule*). The *short budget* rows are the",
        "head ablation on homography_256 (`doc/head_ablation.md` sections 8.22-8.24); H0 and H0-long are scored with",
        "`--model ablation`, H1-H3 with `--model proposed --routing learned`. Every checkpoint holds the whole model",
        "including its backbone.",
        "",
        "| model | file (relative to the repository root) | size | used for |",
        "|---|---|---|---|",
    ]
    for (rel, digest, size), (_, model, use) in zip(rows, FILES):
        lines.append(f"| {model} | `{rel}` | {size / 2**20:.0f} MB | {use} |")
    lines += ["", f"SHA-256 checksums: `{SUMS}`.", ""]
    with open(os.path.join(src_dir, MANIFEST), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def build(out):
    missing = [rel for rel, *_ in FILES if not os.path.isfile(os.path.join(src_dir, rel))]
    if missing:
        sys.exit("missing weight files:\n  " + "\n  ".join(missing))
    rows = []
    for rel, *_ in FILES:
        p = os.path.join(src_dir, rel)
        print(f"[sha256] {rel}")
        rows.append((rel, sha256(p), os.path.getsize(p)))
    write_meta(rows)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".part"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:  # weights barely compress
        for rel in [MANIFEST, SUMS] + [r for r, *_ in FILES]:
            z.write(os.path.join(src_dir, rel), rel)
    with zipfile.ZipFile(tmp) as z:  # read every entry back (CRC check) before the zip replaces an older one
        bad = z.testzip()
        if bad:
            sys.exit(f"zip check failed at {bad}")
    os.replace(tmp, out)
    digest = sha256(out)
    with open(out + ".sha256", "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{digest}  {os.path.basename(out)}\n")
    total = sum(s for _, _, s in rows)
    print(f"[saved] {out} ({os.path.getsize(out) / 2**20:.0f} MB, {len(rows)} weight files, {total / 2**20:.0f} MB)")
    print(f"[sha256] {digest}  {os.path.basename(out)}")
    print(f"[meta] {MANIFEST}, {SUMS} (commit these two)")


def verify():
    sums = os.path.join(src_dir, SUMS)
    if not os.path.isfile(sums):
        sys.exit(f"{SUMS} not found: run this from a checkout of the repository")
    ok = True
    for line in open(sums, encoding="utf-8").read().splitlines():  # LF or CRLF (git's autocrlf on Windows)
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        p = os.path.join(src_dir, rel)
        if not os.path.isfile(p):
            print(f"MISSING  {rel}")
            ok = False
        elif sha256(p) != digest:
            print(f"WRONG    {rel}")
            ok = False
        else:
            print(f"OK       {rel}")
    print("all weights match" if ok else "some weights are missing or differ")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true", help="check the weights in the repository against SHA256SUMS.txt")
    ap.add_argument("--out", default=os.path.join(src_dir, "dist", "hmvcc_final_weights.zip"))
    args = ap.parse_args()
    verify() if args.verify else build(args.out)
