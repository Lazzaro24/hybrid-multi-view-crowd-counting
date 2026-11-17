import torch


class EDCost:
    """Euclidean Distance Cost"""

    def __init__(self) -> None:
        pass

    def __call__(self, pred_coords, target_coords, camera_locations=None):
        pred = pred_coords.clone()
        target = target_coords.clone()

        pred_col = pred.unsqueeze(-2)
        target_row = target.unsqueeze(-3)

        euclidean_dist = torch.sqrt(torch.sum((pred_col - target_row) ** 2, dim=-1) + 1e-8)
        cost = torch.exp(euclidean_dist) - 1.0

        return cost


class MDCost:
    """Mahalanobis Distance Cost with dynamic covariance stretching along camera depth."""

    def __init__(self):
        pass

    def __call__(self, pred_coords, target_coords, camera_locations):
        """
        pred_coords: predicted coordinate points
        target_coords: ground truth coordinate points
        camera_locations: physical 3D camera locations (already normalized)
        """
        pred = pred_coords.clone().squeeze(0)
        target = target_coords.clone().squeeze(0)

        # Empty patch: empty cost matrix
        if target.numel() == 0 or pred.numel() == 0:
            return torch.zeros((1, pred.shape[0], target.shape[0]), device=pred_coords.device)

        # Move the image coordinate origin to the bottom-left corner
        pred[:, 1] = 1.0 - pred[:, 1]
        target[:, 1] = 1.0 - target[:, 1]

        # Remove Z-axis from camera locations to project onto the 2D plane
        cam_locs_2d = camera_locations[:, :-1].float()
        cam_locs_2d[:, 1] = 1.0 - cam_locs_2d[:, 1]

        num_cameras = cam_locs_2d.shape[0]
        mahalanobis_distances = []
        gt_to_cam_distances = []

        for i in range(num_cameras):
            direction_y = target - cam_locs_2d[i]

            # Distance from ground truth to the camera
            dist_to_cam = torch.sqrt(torch.sum(direction_y**2, dim=-1))
            gt_to_cam_distances.append(dist_to_cam)

            dist_min = torch.min(dist_to_cam)
            dist_max = torch.max(dist_to_cam)
            dist_range = dist_max - dist_min

            # Guard against zero range when all points are equidistant or single point
            dist_norm = torch.where(
                dist_range > 1e-6, (dist_to_cam - dist_min) / (dist_range + 1e-8), torch.zeros_like(dist_to_cam)
            )
            dist_exp = torch.exp(dist_norm)

            sigma1 = 1.0 / dist_exp
            sigma2 = 1.0 / dist_exp

            inverse_variance = torch.zeros((dist_exp.shape[0], 2, 2), device=pred_coords.device)
            inverse_variance[:, 0, 0] = 1.0 / sigma1
            inverse_variance[:, 1, 1] = 1.0 / sigma2

            # Rotation: column 1 along the camera-to-target direction, column 2 perpendicular (x, y) -> (-y, x)
            rotation_col1 = direction_y.clone()
            norm = torch.sqrt(rotation_col1[:, 0] ** 2 + rotation_col1[:, 1] ** 2).unsqueeze(-1)
            rotation_col1 = rotation_col1 / (norm + 1e-8)
            rotation_col2 = rotation_col1.clone()
            rotation_col2[:, [0, 1]] = rotation_col2[:, [1, 0]]
            rotation_col2[:, 0] = -1.0 * rotation_col2[:, 0]

            rotation_col1 = rotation_col1.unsqueeze(-1)
            rotation_col2 = rotation_col2.unsqueeze(-1)

            rotation_matrix = torch.cat((rotation_col1, rotation_col2), dim=-1)
            rotation_matrix_t = rotation_matrix.permute(0, 2, 1)

            # Inverse covariance R * Sigma^-1 * R^T
            inverse_covariance = torch.matmul(torch.matmul(rotation_matrix, inverse_variance), rotation_matrix_t)

            pred_col = pred.unsqueeze(-2)
            target_row = target.unsqueeze(-3)
            diff_left = (pred_col - target_row).unsqueeze(-2)
            diff_right = (pred_col - target_row).unsqueeze(-1)

            mahalanobis_dist = torch.sqrt(
                torch.matmul(torch.matmul(diff_left, inverse_covariance), diff_right).squeeze(-1).squeeze(-1) + 1e-8
            )

            mahalanobis_distances.append(mahalanobis_dist)

        mahalanobis_distances = torch.stack(mahalanobis_distances, dim=0)

        # Each GT point uses the cost of its nearest camera
        nearest_camera_indices = torch.stack(gt_to_cam_distances, dim=0).min(0)[1]

        final_mahalanobis = torch.zeros_like(mahalanobis_distances)[0]
        for i in range(nearest_camera_indices.shape[0]):
            cam_idx = nearest_camera_indices[i]
            final_mahalanobis[:, i] = mahalanobis_distances[cam_idx, :, i]

        # Plain Mahalanobis distance (the original's exponential scaling is unstable)
        return final_mahalanobis.unsqueeze(0)
