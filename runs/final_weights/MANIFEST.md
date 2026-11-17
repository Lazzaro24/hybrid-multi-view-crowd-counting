# Final weights of the thesis

Download: LRZ Sync+Share, link in the README (section 4, *Final weights*). Unzip into the repository root; the files land
at the paths below, which are the paths the README's commands use. Check them with
`uv run python scripts/package_final_weights.py --verify` (from the repository root) or `sha256sum -c runs/final_weights/SHA256SUMS.txt`.

The seven final evaluations use the first five files; the two oracle runs reuse the h1_hom and h1_casa
weights (oracle routing on the Stage 4 weights reproduces the Stage 3 oracle exactly, see
`doc/final_evaluations.md` section 7.4). The two baseline *final weights* rows are the last epochs of the same
baseline runs (`doc/results.md`, *Baselines under the final-weights rule*). The *short budget* rows are the
head ablation on homography_256 (`doc/head_ablation.md` sections 8.22-8.24); H0 and H0-long are scored with
`--model ablation`, H1-H3 with `--model proposed --routing learned`. Every checkpoint holds the whole model
including its backbone.

| model | file (relative to the repository root) | size | used for |
|---|---|---|---|
| Baseline (OT) | `runs/baseline/ot/best.pth` | 124 MB | SynMVCrowd baseline after the OT fine-tune; paper protocol, final evaluations |
| Baseline (MSE) | `runs/baseline/run_1/best.pth` | 97 MB | SynMVCrowd baseline trained with MSE (run_1); paper protocol, final evaluations |
| h1_hom | `runs/full_length_dense_mask_homography/4_router_anchor/multiscene_final_stage_4_bud200_homography_256_dout0.7_cw0.01_genloss_a0.7_ro_anchor0.1_fs.pth` | 46 MB | router lineage on homography_256, Stage 4 final weights; learned routing and oracle routing |
| h1_casa | `runs/full_length_dense_mask_final/4_router_anchor/multiscene_final_stage_4_bud200_ca_sa_dout0.7_cw0.01_genloss_a0.7_ro_anchor0.1_fs.pth` | 85 MB | router lineage on the ca_sa attention neck, Stage 4 final weights; learned routing and oracle routing |
| hybrid_hom | `runs/full_length_hybrid/3_sparse_s42/multiscene_final_stage_2_bud200_homography_256_S4D2_s42.pth` | 46 MB | dense-first hybrid (D2 dense + S4D2 sparse expert, seed 42); density rule dzone0.1p |
| J (joint run) | `full_length_joint_homography/1_joint/multiscene_final_stage_joint_hom_s0.01_wr1_wf1.pth` | 46 MB | h1_hom trained in one loop without the curriculum; joint-training comparison |
| H0-control | `runs/full_length_hybrid/4_h0_control/homography_256/last.pth` | 177 MB | fusion + baseline decoder fine-tuned for the hybrid's budget; occlusion and map-quality reference |
| 2D backbone | `runs/train_2d/new_best/resnet18_2d_pretrained_ep4.pth` | 126 MB | ResNet-18 pretrained on per-camera head density maps; initialises every training, needed to load H0-control and the H0 short-budget runs |
| Baseline (OT), final weights | `runs/baseline/ot_frozen/last.pth` | 124 MB | last epoch of the OT fine-tune (epoch 65, the early stop); final-weights comparison |
| Baseline (MSE), final weights | `runs/baseline/run_1/last.pth` | 97 MB | last epoch of run_1 (epoch 200); final-weights comparison |
| H0 (short budget) | `runs/neck_ablation/ablation_v2/homography_256/last.pth` | 178 MB | the 30-epoch homography_256 fusion run with the baseline decoder; Stage 1 of H1-H3, H0 of the head ablation |
| H0-long (short budget) | `runs/head_ablation/homography_256/h0_long/1_decoder/homography_256/last.pth` | 178 MB | H0 trained on for 88 epochs (H1's budget), fusion trainable; budget-matched decoder reference |
| H1 (short budget) | `runs/head_ablation/homography_256/og_mask/4_router_anchor/multiscene_final_stage_4_abl_homography_256_og_dout0.7_cw0.01_genloss_a0.7_ro_anchor0.1_fs.pth` | 46 MB | learned router, pointwise oracle mask, full recipe incl. Stage 4 anchor; learned routing |
| H2 (short budget) | `runs/head_ablation/homography_256/window_mask_D0.4/4_router_anchor/multiscene_final_stage_4_abl_homography_256_W25D0.4_ma50_dout0.7_cw0.01_genloss_a0.7_ro_anchor0.1_fs.pth` | 46 MB | learned router, window-density mask W25 / 0.4 p/m^2 / min area 50 px, H1's recipe; learned routing |
| H3 (short budget) | `runs/head_ablation/homography_256/perf_router/c10_sel/multiscene_final_stage_3_abl_homography_256_unmasked_joint_genloss_ep30_perf_c10_sel.pth` | 46 MB | performance routing, 10 px cells, Stage 3 final weights; learned routing |

SHA-256 checksums: `runs/final_weights/SHA256SUMS.txt`.
