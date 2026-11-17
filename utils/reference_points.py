"""
reference_points.py

compute_reference_points: projects the 3D pillar points of every BEV cell (one per z-anchor) into each camera and
marks which projections are valid. Two checks per z-anchor, a ray failing either is invalid
(doc/camera_bug.md, doc/camera_off_axis.md):

  1. Behind the camera:
       z_cam * z_sign[i] < 1e-2   -> invalid
     z_sign is +1.0 per camera unless a correction is supplied (e.g. from utils/z_sign_corrections_full.json).

  2. Off-axis distortion fold-back:
       r_offaxis > offaxis_threshold[i]   -> invalid
     r_offaxis = tan(angle from the optical axis), from the raw camera-space coordinates (sign-invariant, so no
     z_sign). Beyond it the polynomial distortion model is not monotonic: true field-of-view rays measured
     r <= ~2.85, the confirmed fold-back artifact r >= ~4.64; the threshold 3.0 sits in between (1.2-3.75 gave
     identical results on the camera tested).

Neither check changes the pixel projection itself (image_points / x_norm / y_norm), only the validity mask.
"""

import torch
import numpy as np
import cv2

IMG_SIZE = (720, 1280)  # (H, W)
NUM_Z_ANCHORS = 4
DEFAULT_OFFAXIS_THRESHOLD = 3.0


def compute_reference_points(
    wld_map_paras,
    camera_paras,
    img_size=IMG_SIZE,
    num_z_anchors=NUM_Z_ANCHORS,
    z_sign=None,
    offaxis_threshold=DEFAULT_OFFAXIS_THRESHOLD,
    z_offsets=None,
):
    """
    Args:
        wld_map_paras: [r, a, b, h_actual, w_actual, d_mean, w_min, h_min,
                        w_max, h_max, d_delta] -- unchanged from the original.
        camera_paras: [num_cameras, 15] tensor, rows = [fx,fy,u,v] + distCoeffs(5)
                       + rvec(3) + tvec(3) -- unchanged from the original.
        img_size: (H, W) used to scale normalized intrinsics -- unchanged.
        num_z_anchors: number of vertical BEV pillar samples -- unchanged.
        z_sign: optional list/array, one entry per camera, each +1.0 or -1.0.
                Corrects the sign convention used ONLY by the behind-camera
                validity check (see camera_extrinsics_zsign_bug.md). Defaults
                to +1.0 for every camera (no-op) if not supplied.
        offaxis_threshold: float, OR optional list/array (one entry per
                camera) if some cameras need a different threshold than
                others. Any ray with tan(angle from optical axis) greater
                than this is rejected as a likely distortion fold-back
                artifact (see camera_offaxis_foldback_bug.md). Defaults to
                3.0 for every camera.
        z_offsets: optional sequence of anchor heights relative to d_mean (overrides num_z_anchors). NOTE: d_mean
                is the mean HEAD height of the frame's annotated people (not the ground) and d_delta = 1.75 m, so
                the default linspace(-d_delta, 1.8 + d_delta) + d_mean puts the anchors at feet, head and two points
                above the heads. A feet-to-head range with terrain margins is e.g.
                linspace(-d_delta - 2.5, 2.5, 8) (see cache_reference.py --z_anchors).

    Returns:
        Tensor of shape [num_cameras, h_actual, w_actual, num_z_anchors, 2],
        same shape/semantics as the original function. Invalid points (by
        EITHER criterion) are set to (-10.0, -10.0), same sentinel as before.
    """
    r, a, b, h_actual, w_actual, d_mean, w_min, h_min, w_max, h_max, d_delta = wld_map_paras
    xs = torch.arange(w_actual) / r + w_min - a
    ys = torch.arange(h_actual) / r + h_min - b
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    if z_offsets is not None:
        z_heights = torch.as_tensor(z_offsets, dtype=torch.float32) + d_mean
        num_z_anchors = len(z_heights)
    else:
        z_heights = torch.linspace(-d_delta, 1.8 + d_delta, steps=num_z_anchors) + d_mean
    points_3d = []
    for z in z_heights:
        Z_plane = torch.full_like(grid_x, z.item())
        plane_3d = torch.stack([grid_x, grid_y, Z_plane], dim=-1)
        points_3d.append(plane_3d)

    points_3d = torch.stack(points_3d, dim=-2).reshape(-1, 3)
    points_3d_np = points_3d.numpy().astype(np.float32)

    num_cameras = camera_paras.shape[0]

    if z_sign is None:
        z_sign = [1.0] * num_cameras
    if isinstance(offaxis_threshold, (int, float)):
        offaxis_threshold = [float(offaxis_threshold)] * num_cameras

    ref_points = []
    H, W = img_size

    for i in range(num_cameras):
        cam = camera_paras[i].cpu().numpy()
        fx = cam[0] * W
        fy = cam[1] * H
        cx = cam[2] * W
        cy = cam[3] * H
        dist_coeffs = cam[4:9]
        rvec = cam[9:12]
        tvec = cam[12:15]

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        # --- Pixel projection: UNCHANGED from the original, verified-correct math ---
        image_points, _ = cv2.projectPoints(points_3d_np, rvec, tvec, K, dist_coeffs)
        image_points = image_points.reshape(-1, 2)

        # --- Shared camera-space coordinates, used by BOTH validity checks below ---
        R_matrix, _ = cv2.Rodrigues(rvec)
        points_cam = (R_matrix @ points_3d_np.T).T + tvec
        X_cam, Y_cam, Z_cam_raw = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]

        # --- Check 1: behind-camera (Z-sign corrected) ---
        z_cam_corrected = Z_cam_raw * z_sign[i]
        behind_camera_invalid = z_cam_corrected < 1e-2

        # --- Check 2: off-axis distortion fold-back (NEW) ---
        # Sign-invariant: X/Z and Y/Z are unaffected by a uniform sign flip of
        # (X, Y, Z), so no z_sign correction is needed here.
        eps = 1e-6
        safe_Z = np.where(np.abs(Z_cam_raw) < eps, eps, Z_cam_raw)
        x_prime = X_cam / safe_Z
        y_prime = Y_cam / safe_Z
        r_offaxis = np.sqrt(x_prime**2 + y_prime**2)
        offaxis_invalid = r_offaxis > offaxis_threshold[i]

        invalid_mask = behind_camera_invalid | offaxis_invalid

        x_norm = (image_points[:, 0] / W) * 2.0 - 1.0
        y_norm = (image_points[:, 1] / H) * 2.0 - 1.0

        x_norm[invalid_mask] = -10.0
        y_norm[invalid_mask] = -10.0

        uv_norm = torch.stack(
            [torch.tensor(x_norm, dtype=torch.float32), torch.tensor(y_norm, dtype=torch.float32)], dim=-1
        )

        uv_norm = uv_norm.view(int(h_actual), int(w_actual), num_z_anchors, 2)
        ref_points.append(uv_norm)

    return torch.stack(ref_points, dim=0)
