# Hybrid Multi-View Crowd Counting

Source code of the master's thesis *Hybrid Multi-View Crowd Counting* (Lazzaro Kaçani, Hochschule München, 2026).
The model fuses synchronized camera views onto a bird's-eye-view (BEV) ground plane and counts people there. It has
three parts: a frozen 2D backbone (dilated ResNet-18), a fusion **neck** (homography projection or BEVFormer-style
deformable cross-attention), and a **head**: either a single density decoder or a hybrid of a dense expert (crowds)
and a sparse expert (isolated people) with a routing decision.

This README is the manual: it explains how to set up the environment, prepare the data, train the baseline and the
proposed models, and run the evaluations. The results and their discussion are in the thesis.

**Contents**

1. [Environment](#1-environment)
2. [Datasets](#2-datasets)
3. [Training](#3-training)
4. [Evaluation](#4-evaluation)
5. [Further experiments](#5-further-experiments)
6. [Repository layout](#6-repository-layout)
7. [License and citation](#7-license-and-citation)

---

## 1. Environment

Every command in this README runs from the repository root.

Requirements: Python 3.12, an NVIDIA GPU with CUDA 12.x, and [uv](https://docs.astral.sh/uv/). The long trainings
and the paper-protocol evaluations ran on a machine with a 24 GB GPU; the occlusion, map-quality and cost
evaluations and the figures of the final models ran on a 12 GB RTX 3080 Ti.

```powershell
# install uv once (Windows; on Linux/macOS: curl -LsSf https://astral.sh/uv/install.sh | sh)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

uv sync --python 3.12          # creates .venv from uv.lock
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

- On Windows, `torch`/`torchvision` come from the PyTorch CUDA 12.1 index (configured in `pyproject.toml`). On Linux
  they come from PyPI.
- The OT losses that the recipes use run GeomLoss with the `tensorized` backend. KeOps warnings at startup
  ("No C++ compiler found", "Switching to CPU only") can be ignored.
- Run scripts with `uv run python <script>`, or activate `.venv` first.
- Linting and formatting: `uv run ruff check .` and `uv run ruff format .` (settings in `pyproject.toml`).
- `Dockerfile` builds a Linux CUDA image with the same environment (`docker build -t hmvcc .`).

### Launchers

Every long run of the thesis was started on a Windows machine through the PowerShell launchers in `remote_scripts/`.
A launcher starts the job as a detached process (it keeps running after the SSH session closes), writes logs to
`logs/`, and prints commands to follow and stop the job. The chain launchers (`run_remote_budget_*`,
`run_remote_ot_stage_*`, `run_remote_joint_h1`, `run_camera_ablation`, `run_camera_ablation_train`,
`run_cross_dataset`, `run_final_evals`) also accept:

- `-DryRun`: print the numbered steps and the exact `uv run python ...` command of each step, run nothing. Use this
  on Linux, or to run one step by hand.
- `-StartStep N` (`-Only <names>` in `run_final_evals`): resume a chain at step `N` after a crash.

`run_camera_ablation_all.ps1` runs the whole camera ablation and takes no options: rerunning it after a crash skips
the finished parts. Every launcher describes its runs, budgets and outputs in its header comment.

**Where the outputs end up.** A chain writes everything into `<OutDir>/` (e.g. `full_length_hybrid/`). For the
thesis, the finished chains were then moved to `runs/<OutDir>/`; only the joint run stayed in
`full_length_joint_homography/`. The paths below, and the final-weights download, use that layout. The evaluation
launchers that score the finished models (`run_final_evals.ps1`, `run_camera_ablation_all.ps1`,
`run_cross_dataset.ps1`) name the weights as they lay before the move (`full_length_*\...` directly in the root):
copy the chain folders there, or prefix those paths with `runs\`.

---

## 2. Datasets

### SynMVCrowd (training and every reported result)

Download the SynMVCrowd release (Zhang et al., IJCV 2026, [code and data](https://github.com/zqyq/SynMVCrowd),
[Dropbox](https://www.dropbox.com/scl/fo/1kkx2rexita03zrh97nli/ANMJ2Q8fI2PIOrqSHiN3heA?rlkey=pqio574ua37awablg3ffaitu8&st=ie89yfel&dl=0))
and put it under `datasets/SynMVCrowd/data/` (git-ignored). Layout:

```
datasets/SynMVCrowd/data/
├── train/      30 scenes
├── validate/   10 scenes
└── test/       10 scenes
    └── scene_<id>/
        ├── <snapshot>/            0, 1, 2, ... one folder per frame
        │   ├── jpgs/<cam>.jpg     one image per camera
        │   └── jsons/<cam>.json   calibration, wld_map_paras, head annotations
        └── master_refs.pt         reference-point cache, created below
```

Files that come with the repository and are read automatically:

| file | purpose |
|---|---|
| `datasets/SynMVCrowd/optimal_cameras_filtered.json` | the fixed K-means cameras per scene (proposed trainer, map-quality evaluation) |
| `utils/excluded_scenes.json` | `scene_52` and `scene_71` are left out, so 28 train scenes are used |
| `utils/excluded_frames.json` | single frames left out per scene (corrupted geometry) |
| `utils/z_sign_corrections_full.json` | per-camera sign fix for the reference-point projection |

**Reference-point cache.** The attention necks and the proposed trainer need the projected reference points of all
50 cameras of every scene. Build them once (scenes that already have the file are skipped):

```powershell
uv run python utils/cache_reference.py                                         # master_refs.pt (4 height anchors), used by every training
uv run python utils/cache_reference.py --name master_refs_k8.pt --z_anchors 8  # only for the hca / mvha necks in the neck ablation
```

The training chains below run the first command as their step 1, so it is not needed before them.

### Wildtrack and MultiviewX (cross-dataset check)

Download the two datasets and put them under their (git-ignored) `data/` folders:

| dataset | download | folder | contents |
|---|---|---|---|
| Wildtrack | [EPFL CVLab](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/) | `datasets/Wildtrack/data/` | `calibrations/`, `ds/`, … |
| MultiviewX | [Kaggle](https://www.kaggle.com/datasets/mrriandmstique/multiview-x-multi-camera-tracking-3d) | `datasets/MultiviewX/data/` | `Image_subsets/`, `annotations_positions/`, `calibrations/`, … |

They are used only to test how the SynMVCrowd-trained models transfer, see [5. Further experiments](#5-further-experiments).
Both loaders (`datasets/multiview_detection.py`) compute their reference points on the fly, so no cache is needed.

---

## 3. Training

All models share one budget unit: **one iteration = one frame, 5 views, 3 random 200 × 200 BEV patches, batch 1**.
Baseline (Gaussian) trains for 200 epochs × 1,400 iterations = **280k iterations**. The budget-matched proposed
chains use the same total.

**Model names.** The launchers, folders and this README use run names; the thesis names the models after their neck
and head:

| thesis | run name |
|---|---|
| Baseline (Gaussian) | `run_1` (MSE on Gaussian density maps) |
| Baseline (OT) | `ot` |
| Attention-Hybrid (the proposed model) | h1_casa |
| Homography-Hybrid | h1_hom |
| Homography-Hybrid (joint) | J |
| Homography-Rule | hybrid_hom |
| Homography-Decoder, Homography-Decoder (OT) | H0-control, H0-OT |
| Head ablation at the short budget: Decoder, Decoder (long), Pointwise mask, Window mask, Performance-based | H0, H0-long, H1, H2, H3 |

### 3.0 2D backbone pretraining (shared by every model)

ResNet-18 (ImageNet), stride 8, trained on per-camera head density maps, then frozen in every model:

```powershell
.\remote_scripts\run_remote_2d_pretrain.ps1
# or: uv run python training_scripts/train_2d_pretrain.py --epochs 15 --lr 1e-4
```

The script saves a checkpoint after every epoch to `runs/train_2d/<timestamp>/resnet18_2d_pretrained_ep<N>.pth`
and copies the epoch with the **lowest val MAE** to `best.pth` in the same folder. Every model uses that checkpoint.
The log (`logs\pretrain_2d_log.txt`) prints the best epoch so far after every epoch and the final pick at the end.

Copy `best.pth` to the path that every trainer and chain uses by default. That file name is fixed as `..._ep4.pth`
in the defaults, so keep it even if a different epoch is the best:

```powershell
mkdir runs\train_2d\new_best
copy runs\train_2d\<timestamp>\best.pth runs\train_2d\new_best\resnet18_2d_pretrained_ep4.pth
```

### 3.1 Baseline (reference): SynMVCrowd homography baseline

`training_scripts/train_baseline.py`. Re-implementation of the SynMVCrowd baseline (homography projection, view
selection, decoder), with a frozen backbone. Results go to `runs/baseline/<run_name>/` (`best.pth`, `last.pth`).

```powershell
# Baseline (Gaussian), "run_1": SGD 1e-4, warmup + cosine, 200 epochs, from scratch
.\remote_scripts\run_remote_baseline_train.ps1 -Backbone runs\train_2d\new_best\resnet18_2d_pretrained_ep4.pth `
    -Loss mse -Optimizer sgd -Lr 1e-4 -LrSchedule cosine -InitFrom "" -Epochs 200 -Patience 0 -RunName run_1

# Baseline (OT): the paper's OT fine-tune of run_1 (Adam 1e-5, constant, early stop after 25 epochs)
.\remote_scripts\run_remote_baseline_train.ps1 -Backbone runs\train_2d\new_best\resnet18_2d_pretrained_ep4.pth `
    -InitFrom runs\baseline\run_1\best.pth -RunName ot
```

Add `-Resume` to continue `runs\baseline\<RunName>\last.pth`, or `-TrainBackbone` to train the backbone jointly as
the released code does (not used for the reported numbers). Run without the launcher:
`uv run python training_scripts/train_baseline.py --backbone <...> --run_name run_1 --loss mse --val_every 5`.

### 3.2 Proposed head: learned router (Attention-Hybrid, Homography-Hybrid)

The proposed head of the thesis (the "router lineage"; H1 in the head ablation). A dense and a sparse expert are
trained on an oracle (ground-truth density) mask, then a learned router chooses between them per BEV cell. Trainer:
`training_scripts/train_proposed_multiscene.py` (settings in `training_scripts/proposed_trainer/config.py`, `--help`
lists every flag). The full budget-matched chain runs in one launcher:

```powershell
# h1_hom (Homography-Hybrid): homography_256 fusion
.\remote_scripts\run_remote_budget_ca_sa_h1.ps1 -Neck homography_256 -OutDir full_length_dense_mask_homography
# h1_casa (Attention-Hybrid): the same recipe on the ca_sa attention fusion
.\remote_scripts\run_remote_budget_ca_sa_h1.ps1 -Neck ca_sa -OutDir full_length_dense_mask
```

Chain steps (50 + 50 + 25 + 50 + 25 = 200 epochs × 1,400 iterations). Each stage hands its **final** weights to the
next:

| step | what | script |
|---|---|---|
| 1 | reference-point cache | `utils/cache_reference.py` |
| 2 | fusion (neck + baseline decoder, EMA), 50 epochs; this is also the H0 reference | `training_scripts/neck_ablation.py` |
| 3 | turn the fusion into a Stage 1 checkpoint | `training_scripts/make_stage1_from_ablation.py` |
| 4 | Stage 2: both experts, masked MSE on the oracle zone | `train_proposed_multiscene.py --stage 2` |
| 5 | OT step: fresh sparse expert with the generalized OT loss | `... --stage 2 --sparse_only --reinit_sparse --sparse_loss genloss` |
| 6 | Stage 3: router only, Tversky loss (α 0.7) on the oracle mask | `... --stage 3` |
| 7 | Stage 4: router only, fused-map loss + shape anchor 0.1 | `... --stage 4 --router_only` |
| 8–10 | paper-protocol test of Stage 4 final, Stage 4 best, Stage 3 with oracle routing | `evaluation/evaluate_paper_protocol.py` |

Everything lands in `<OutDir>/` (`1_fusion/`, `2_dense_mask/`, `2_ot_sparse/`, `3_router/`, `4_router_anchor/`,
`eval/`, `chain_log.txt`). In the thesis, h1_hom is `runs/full_length_dense_mask_homography/` and h1_casa is
`runs/full_length_dense_mask_final/`: a redo from the OT step on the Stage 2 weights of `full_length_dense_mask/`,
handing on final weights (the command is in the launcher's header). The trainer also loads the older stride-4
checkpoints of the early experiments (it detects the layout from the checkpoint); their launchers are
`run_remote_stage123_stride8.ps1` and related scripts.

### 3.3 Density-rule variant (Homography-Rule, no router)

A tested variant of the proposed head: a fixed density rule replaces the learned router, and the sparse expert is
deeper. The steps run one after the other, each on frozen weights of the previous one:

1. **Fusion** (homography_256), the same as step 2 above. Its decoder is **H0**.
2. **D2:** dense expert + adapter on the frozen fusion, MSE only, EMA.
3. **S4D2:** a deep sparse expert (genloss + empty-ground penalty) on the frozen D2.
4. **Routing:** a fixed, ground-truth-free density rule. The dense expert is used where its own predicted density
   is at least a threshold, sparse peaks elsewhere. The threshold is chosen by **val GAME3**.

```powershell
# budget-matched chain (50 + 60 + 90 = 200 epochs) + H0-control (Homography-Decoder: decoder fine-tuned for the same budget)
.\remote_scripts\run_remote_budget_homography_hybrid.ps1 -WithVal
# then the baseline's OT fine-tune on both arms, H0-OT (Homography-Decoder (OT)) and hybrid-OT (needs the weights of the chain above in the same OutDir)
.\remote_scripts\run_remote_ot_stage_homography_hybrid.ps1 -WithVal
```

Main options: `-Seeds 42,7` (head seeds; the fusion is trained once), `-Neck`, `-OutDir` (default
`full_length_hybrid`), `-SkipControl`. The output goes to `full_length_hybrid/` (thesis run:
`runs/full_length_hybrid/`, seed 42): `1_fusion/`, `2_dense_s<seed>/`, `3_sparse_s<seed>/`, `4_h0_control/`,
`5_h0_ot/`, `6_hybrid_ot_s<seed>/`, and `eval/`, which holds the density sweeps, the selected threshold
`routing_s<seed>.txt` and the paper-protocol JSONs. The short-budget versions (30-epoch fusion, 2 seeds) are in
`run_remote_dense_recipe.ps1`.

### 3.4 Joint training (Homography-Hybrid, joint)

The control for the staged training of 3.2: the same h1_hom model (fresh homography_256 fusion, both experts, router)
trained in **one loop** for the same 280k iterations. The loss is the sum of the stage losses: Stage 2 masked MSE for
the dense expert, the OT step's sparse loss × 0.01, Tversky for the router and the fused-map MSE (weights 1).
Gradients flow end to end; final weights, no EMA.

```powershell
.\remote_scripts\run_remote_joint_h1.ps1
```

Steps: 1 smoke run (2 epochs, checks the balance of the four loss terms), 2 joint training
(`train_proposed_multiscene.py --joint --neck ablation`), 3–4 paper-protocol test with learned and oracle routing.
`-StopAfter 1` runs only the smoke step; `-ResumeEpoch N` restarts step 2 from its epoch-`N` checkpoint after a crash.
The output goes to `full_length_joint_homography/` (`0_smoke/`, `1_joint/`, `eval/`, `chain_log.txt`).

---

## 4. Evaluation

### Final weights

The weights of the final models are not in git. Download them from LRZ Sync+Share:
**[hmvcc_final_weights.zip](https://syncandshare.lrz.de/getlink/fiJK3T7ybQGmAnKzXwpAB4/hmvcc_final_weights.zip)**
(1.46 GB, link valid until December 2027, SHA-256
`c39e5bddfec39902148184baca75a105a40a376364f42ec77e3c6efc3972f001`). Unzip the archive into the repository root.
The files land at the paths the commands below use, and a check confirms them:

```powershell
curl.exe -L -o hmvcc_final_weights.zip https://syncandshare.lrz.de/dl/fiJK3T7ybQGmAnKzXwpAB4/hmvcc_final_weights.zip
tar -xf hmvcc_final_weights.zip      # Windows tar.exe; on Linux: unzip hmvcc_final_weights.zip
uv run python scripts/package_final_weights.py --verify
```

| model | file |
|---|---|
| Baseline (OT) | `runs/baseline/ot/best.pth` |
| Baseline (Gaussian) | `runs/baseline/run_1/best.pth` |
| h1_hom | `runs/full_length_dense_mask_homography/4_router_anchor/multiscene_final_stage_4_..._fs.pth` |
| h1_casa | `runs/full_length_dense_mask_final/4_router_anchor/multiscene_final_stage_4_..._fs.pth` |
| hybrid_hom | `runs/full_length_hybrid/3_sparse_s42/multiscene_final_stage_2_bud200_homography_256_S4D2_s42.pth` |
| J (joint run) | `full_length_joint_homography/1_joint/multiscene_final_stage_joint_hom_s0.01_wr1_wf1.pth` |
| H0-control | `runs/full_length_hybrid/4_h0_control/homography_256/last.pth` |
| 2D backbone | `runs/train_2d/new_best/resnet18_2d_pretrained_ep4.pth` |
| Baseline (OT), final weights | `runs/baseline/ot_frozen/last.pth` |
| Baseline (Gaussian), final weights | `runs/baseline/run_1/last.pth` |
| H0 (short budget) | `runs/neck_ablation/ablation_v2/homography_256/last.pth` |
| H0-long (short budget) | `runs/head_ablation/homography_256/h0_long/1_decoder/homography_256/last.pth` |
| H1 (short budget) | `runs/head_ablation/homography_256/og_mask/4_router_anchor/multiscene_final_stage_4_..._fs.pth` |
| H2 (short budget) | `runs/head_ablation/homography_256/window_mask_D0.4/4_router_anchor/multiscene_final_stage_4_..._fs.pth` |
| H3 (short budget) | `runs/head_ablation/homography_256/perf_router/c10_sel/multiscene_final_stage_3_..._perf_c10_sel.pth` |

The full file names and checksums are in `runs/final_weights/` (`MANIFEST.md`, `SHA256SUMS.txt`). The two oracle
evaluations reuse the h1_hom and h1_casa weights. H0 and H0-long are evaluated with `--model ablation`, H1–H3 with
`--model proposed --routing learned`. `scripts/package_final_weights.py` (without `--verify`) builds the archive
from a checkout that has the weights.

### 4.1 Paper protocol

`evaluation/evaluate_paper_protocol.py` scores every model the same way as the SynMVCrowd paper and its released
test code:

- every frame of the split (test: 2,062 frames) × 5 random draws of 5 views = 10,310 samples, all seeded;
- counting: MAE, MSE, NAE;
- localization: min-max normalised map > 0.2, NMS 2.5 px, Hungarian matching within 0.5 m, giving MODA, MODP,
  precision, recall and F1.

Results are written to `eval_<split>_<weights>[_<routing>].json` next to the weights (or to `--out`). One test run
takes under an hour and less than 4 GB of GPU memory. The chains above already run it for their own models.
Commands for the final weights:

```powershell
# Baselines (--model baseline)
uv run python evaluation/evaluate_paper_protocol.py --model baseline --weights runs/baseline/run_1/best.pth
uv run python evaluation/evaluate_paper_protocol.py --model baseline --weights runs/baseline/ot/best.pth

# Learned-router head (--model proposed), Homography-Hybrid: learned router, or the GT mask in its place (routing ceiling)
uv run python evaluation/evaluate_paper_protocol.py --model proposed --routing learned `
    --weights runs/full_length_dense_mask_homography/4_router_anchor/multiscene_final_stage_4_bud200_homography_256_dout0.7_cw0.01_genloss_a0.7_ro_anchor0.1_fs.pth
uv run python evaluation/evaluate_paper_protocol.py --model proposed --routing oracle `
    --weights runs/full_length_dense_mask_homography/4_router_anchor/multiscene_final_stage_4_bud200_homography_256_dout0.7_cw0.01_genloss_a0.7_ro_anchor0.1_fs.pth
#   oracle on the Stage 4 weights = the Stage 3 oracle exactly (Stage 4 trains the router only)
#   Attention-Hybrid (h1_casa): the same with runs/full_length_dense_mask_final/4_router_anchor/multiscene_final_stage_4_bud200_ca_sa_..._fs.pth

# Density-rule variant, Homography-Rule: the density rule (threshold from the val sweep, here 0.1 p/m^2; "p" = sparse as peaks)
uv run python evaluation/evaluate_paper_protocol.py --model proposed --routing dzone0.1p `
    --weights runs/full_length_hybrid/3_sparse_s42/multiscene_final_stage_2_bud200_homography_256_S4D2_s42.pth
#   or read the threshold file the sweep wrote:  --routing @runs/full_length_hybrid/eval/routing_s42.txt
#   dense expert alone:                          --routing dense --weights runs/full_length_hybrid/2_dense_s42/multiscene_final_stage_2_bud200_homography_256_D2_s42.pth

# Baseline decoder on a neck-ablation fusion (H0; H0-control = Homography-Decoder; H0-OT): --model ablation
uv run python evaluation/evaluate_paper_protocol.py --model ablation --weights runs/full_length_hybrid/4_h0_control/homography_256/last.pth

# Homography-Hybrid (joint), run J
uv run python evaluation/evaluate_paper_protocol.py --model proposed --routing learned `
    --weights full_length_joint_homography/1_joint/multiscene_final_stage_joint_hom_s0.01_wr1_wf1.pth
```

Useful flags: `--split validate`, `--num_views 3|7|10` (camera ablation), `--view_norm`, `--per_scene` (saves after
every scene, so a crashed run can continue), `--num_workers`. To run an evaluation detached:
`.\remote_scripts\run_remote_eval_paper_protocol.ps1 -Model baseline -Weights runs\baseline\run_1\best.pth`.

### 4.2 Final evaluations (intervals, zones, weather, thresholds, router, occlusion, cost, figures)

The numbers of the thesis come from one last evaluation pass over the final models (E1–E8):

```powershell
# E1/E4: paper protocol of the 7 final evaluations (baselines, h1_hom, h1_casa, hybrid_hom, both oracles) with one row
# per sample (--dump_samples) and a detection-threshold sweep (--cls_thres_list)
.\remote_scripts\run_final_evals.ps1                       # -> runs/final_evals/<name>.json, <name>.samples.csv

# E1-E3, E5: scene and frame bootstrap intervals, paired differences, density zones, weather, router (no GPU)
uv run python evaluation/analyze_samples.py --dir runs/final_evals      # -> runs/final_evals/analysis.json

# E6: occlusion recovery and map quality of the final models
uv run python evaluation/eval_occlusion.py --models bud200_h0ctrl:homography_256 `
    --proposed h1_hom=<checkpoint>:learned --baselines ot=runs/baseline/ot/best.pth
# E7: parameters, FLOPs, memory and time per model
uv run python evaluation/model_cost.py
# E8: qualitative figures (input views + map of every model) for a sparse, a dense and an occluded frame
uv run python scripts/visualisation/plot_final_models.py --pick dense
```

**Map-quality frames and paired comparisons.** Besides the paper protocol, map quality (far mass, GAME, per-person
recovery) is measured on fixed K-means cameras, 50 frames per held-out scene, with
`evaluation/eval_map_quality.py`. Two models are compared per scene with bootstrap intervals using
`evaluation/compare_map_quality.py`.

The figure scripts in `scripts/visualisation/` read the weights and result files of the runs above and write to
`figs/` or `diagnostics/` (both git-ignored).

---

## 5. Further experiments

| experiment | entry point |
|---|---|
| Neck ablation (RQ1): homography_256, ca, sa_ca, ca_sa, bev3, hca, mvha with the same decoder, loss and budget | `.\remote_scripts\run_neck_ablation.ps1 -RunName ablation_v2` → `training_scripts/neck_ablation.py` (summary: `--run_name <run> --summarize`) |
| Head ablation (RQ2): routing variants, sparse-expert variants, dense recipe | `run_head_ablation_arm.ps1`, `run_remote_ablation_neck_routing.ps1`, `run_remote_sparse_expert_s012.ps1`, `run_remote_dense_recipe.ps1` |
| Camera count: 3/5/7/10 views, at test time (tier B) and trained at that view count (tier A) | `.\remote_scripts\run_camera_ablation_all.ps1` → `training_scripts/camera_ablation.py`; tables: `uv run python evaluation/summarize_camera_ablation.py` |
| Cross-dataset: Wildtrack and MultiviewX, zero-shot and fine-tuned on 10% / 100% of their train split (3,600 steps, backbone trainable; seeds 0–2 for baseline_ot and h1_casa at 100%) | `.\remote_scripts\run_cross_dataset.ps1` (`-Parts check,zeroshot,finetune`, `-Seeds`, `-Only`) → `evaluation/evaluate_cross_dataset.py`, `training_scripts/finetune_cross_dataset.py`; results in `runs/cross_dataset/{check,zero_shot,finetune}/` |
| Joint training vs staged training | see [3.4](#34-joint-training-homography-hybrid-joint) |

The diagnostics behind the design decisions (expert, router and density-rule analyses, threshold sweeps, zone error
decomposition, camera geometry) are in `scripts/diagnosis/`; each script explains its inputs and outputs in its
docstring.

---

## 6. Repository layout

| path | contents |
|---|---|
| `models/` | `backbone/` (dilated ResNet-18), `neck/baseline/` (homography projector, view selection), `neck/proposed/` (spatial cross-attention, deformable attention, BEV encoder, multi-view height attention), `head/baseline/` (decoder), `head/proposed/routing.py` (dense / sparse experts, router) |
| `losses/` | Gaussian MSE, OT losses (`ot_loss/genloss_synmvcrowd.py` = the baseline's, `genloss.py`, `sparse_ot_loss.py`), vendored GeomLoss |
| `datasets/` | the SynMVCrowd dataset class and K-means camera selection JSONs, the Wildtrack / MultiviewX loaders (`multiview_detection.py`), and the (git-ignored) `data/` folders |
| `training_scripts/` | trainers: `train_2d_pretrain.py`, `train_baseline.py`, `neck_ablation.py`, `train_proposed_multiscene.py` (entry point for the training stages and `--joint`) + `proposed_trainer/` (config, CLI, model, losses, targets, validation), `camera_ablation.py`, `finetune_cross_dataset.py`, `make_stage1_from_ablation.py` |
| `evaluation/` | `evaluate_paper_protocol.py` (headline numbers), `evaluate_cross_dataset.py`, `analyze_samples.py` (intervals and breakdowns of the per-sample results), `model_cost.py`, map-quality, occlusion, per-frame, view-dropout and train-bias evaluations, `compare_map_quality.py`, `summarize_camera_ablation.py`, `metrics/` |
| `remote_scripts/` | PowerShell launchers for every training and evaluation chain (see [Launchers](#launchers)) |
| `utils/` | reference-point cache and projection, scene/frame exclusions, oracle mask, camera-slot handling, calibration and off-axis checks, weather index |
| `scripts/` | analysis tools: `diagnosis/`, `visualisation/`, `verification/` (exclusion-list check), `package_final_weights.py` |
| `runs/final_weights/` | manifest and checksums of the final weights; every run writes its outputs under `runs/` (git-ignored) |
| `Dockerfile` | CUDA image with the uv environment |

Some docstrings and comments refer to `doc/*.md` files. These are the author's research notes, which are not part of
this repository; the thesis contains their final content.

---

## 7. License and citation

The code is licensed under the [Apache License 2.0](LICENSE). It contains code derived from
[SynMVCrowd](https://github.com/zqyq/SynMVCrowd) (Apache 2.0), [GeomLoss](https://github.com/jeanfeydy/geomloss)
(MIT) and [torchvision](https://github.com/pytorch/vision) (BSD 3-Clause); see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The datasets are not included and have their own terms of use.

If you use this code, please cite the thesis:

```bibtex
@mastersthesis{Kacani_2026_HybridMVCC,
    author = {Lazzaro Ka{\c{c}}ani},
    title  = {Hybrid Multi-View Crowd Counting},
    school = {Hochschule M{\"u}nchen University of Applied Sciences},
    year   = {2026}
}
```

and the SynMVCrowd benchmark it builds on:

```bibtex
@article{Zhang_2026_SynMVCrowd,
    author  = {Q. Zhang and D. Chen and Y. Gong and H. Huang},
    title   = {{SynMVCrowd}: A Large Synthetic Benchmark for Multi-view Crowd Counting and Localization},
    journal = {International Journal of Computer Vision},
    year    = {2026},
    volume  = {134},
    pages   = {191},
    doi     = {10.1007/s11263-026-02778-3}
}
```
