# Introduction
This repository is forked from https://github.com/ultralytics/ultralytics (YOLOv8-Pose). The original keypoints detection loss is the weighted sum of the distance between the predicted keypoints and the ground truth, which treats each point independently. Instead, this repo proposes a geometric consistency loss which adds two loss terms: the distance between the center and the difference in angles of three sides of the triangle composed of three keypoints, so as to consider geometric consistency and evaluate the performance of keypoints detection network better during training, which improves the accuracy of keypoints detection network.

Note: The new loss function can be only used for the specific task, which each object contain 3 keypoints.
# Core Code
Line 114 in ultralytics/utils/loss.py
```
class TriangleLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, tri_weight) -> None:
        super().__init__()
        self.tri_weight = tri_weight

    def forward(self, pred_kpts, gt_kpts):
        vis_gt_kpts = gt_kpts[torch.all(gt_kpts[:, :, 2] > 0, dim=1)]
        vis_pred_kpts = pred_kpts[torch.all(gt_kpts[:, :, 2] > 0, dim=1)]
        scale_gt_kpts = vis_gt_kpts[:, :, :2] / vis_gt_kpts[:, :, 2].unsqueeze(2)
        scale_pred_kpts = vis_pred_kpts[:, :, :2] / vis_gt_kpts[:, :, 2].unsqueeze(2)
        gt_centers = torch.mean(scale_gt_kpts, dim=1)
        pred_centers = torch.mean(scale_pred_kpts, dim=1)
        loss_centers = ((gt_centers - pred_centers) ** 2).mean()

        ab_diff = (scale_gt_kpts[:, 1, :] - scale_gt_kpts[:, 0, :]) - (scale_pred_kpts[:, 1, :] - scale_pred_kpts[:, 0, :])
        bc_diff = (scale_gt_kpts[:, 2, :] - scale_gt_kpts[:, 1, :]) - (scale_pred_kpts[:, 2, :] - scale_pred_kpts[:, 1, :])
        ca_diff = (scale_gt_kpts[:, 0, :] - scale_gt_kpts[:, 2, :]) - (scale_pred_kpts[:, 0, :] - scale_pred_kpts[:, 2, :])
        ab_length = torch.norm(scale_gt_kpts[:, 1, :] - scale_gt_kpts[:, 0, :], dim=1, keepdim=True)
        bc_length = torch.norm(scale_gt_kpts[:, 2, :] - scale_gt_kpts[:, 1, :], dim=1, keepdim=True)
        ca_length = torch.norm(scale_gt_kpts[:, 0, :] - scale_gt_kpts[:, 2, :], dim=1, keepdim=True)
        ab_diff_n = torch.abs(ab_diff / ab_length)
        bc_diff_n = torch.abs(bc_diff / bc_length)
        ca_diff_n = torch.abs(ca_diff / ca_length)
        loss_diff = (ab_diff_n + bc_diff_n + ca_diff_n).mean()

        return self.tri_weight[0] * loss_centers + self.tri_weight[1] * loss_diff
