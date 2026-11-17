"""
utils/camera_slots.py

Run a model trained with one camera count on another (camera ablation, RQ3).

The homography fusions (baseline, homography_256) and mvha have no per-camera weights: they max / attend over
whatever cameras they get. The deformable cross-attentions (DeformableCrossAttention of the legacy / ca_sa / sa_ca
necks, MultiHeadSpatialCrossAttention of BEVFormer / hca) predict sampling offsets and weights with one block of
outputs per camera slot (offset_predictor / weight_predictor, out_features = num_cameras * per_slot).

The trainers draw the cameras with random.sample, so every camera has passed through every slot -- but the slots
are NOT the same function: in the ca_sa fusion the per-slot anchor weights spread by ~0.11 around their mean of 0.25
and the offsets by as much as their own size (each slot kept its own random init). So a slot-mean would be a
sampling pattern no camera was ever trained with. Instead camera i uses trained slot i mod n: with fewer cameras
the model is exactly the trained one with some cameras dropped, with more every camera still gets a trained slot.

Two fusions also scale with the camera count itself (renormalise_views undoes that):
- ViewFeatureSelector (baseline, homography_256) softmaxes a per-pixel score ACROSS the cameras, so the mean weight is
  1/N and the features (and the count) shrink like 5/N -> weights * N/5.
- DeformableCrossAttention (ca_sa) sums the cameras' samples, so the fused feature grows like N/5 -> sum * 5/N.
MultiHeadSpatialCrossAttention already divides by the number of cameras that see the cell.
"""

import torch
import torch.nn as nn

SLOT_LAYERS = ("offset_predictor", "weight_predictor")


def slot_spread(model):
    """Per layer: mean over output rows of the std across slots, relative to the mean |weight| (0 = identical slots)."""
    out = {}
    for name, module in model.named_modules():
        n = getattr(module, "num_cameras", None)
        for layer in SLOT_LAYERS:
            lin = getattr(module, layer, None) if n else None
            if isinstance(lin, nn.Linear):
                w = lin.weight.detach().float().view(n, -1, lin.in_features)
                out[f"{name}.{layer}"] = float(w.std(0).mean() / w.abs().mean().clamp(min=1e-12))
    return out


@torch.no_grad()
def set_camera_count(model, num_cameras):
    """Resize every per-camera-slot layer of `model` to `num_cameras` slots, slot i = trained slot i mod n.
    Returns the names of the resized modules. A no-op (empty list) when the count already matches."""
    resized = []
    for name, module in model.named_modules():
        n = getattr(module, "num_cameras", None)
        if not n or n == num_cameras:
            continue
        layers = [attr for attr in SLOT_LAYERS if isinstance(getattr(module, attr, None), nn.Linear)]
        module.num_cameras = num_cameras  # without slot layers (mvha) the count is only stored, no weight depends on it
        if not layers:
            continue
        slots = torch.arange(num_cameras) % n
        for layer in layers:
            old = getattr(module, layer)
            per_slot = old.out_features // n
            assert per_slot * n == old.out_features, f"{name}.{layer}: {old.out_features} outputs, {n} cameras"
            new = nn.Linear(old.in_features, per_slot * num_cameras, bias=old.bias is not None).to(old.weight)
            new.weight.copy_(old.weight.view(n, per_slot, -1)[slots].reshape(-1, old.in_features))
            if old.bias is not None:
                new.bias.copy_(old.bias.view(n, per_slot)[slots].reshape(-1))
            setattr(module, layer, new)
        resized.append(name)
    return resized


def renormalise_views(model, num_cameras, trained_cameras=5):
    """Restore the trained per-camera magnitude for num_cameras views (see the module docstring).
    Returns {module name: scale} of the modules it changed."""
    from models.neck.baseline.feature_selection import ViewFeatureSelector
    from models.neck.proposed.deformable_attention import DeformableCrossAttention

    scales = {}
    for name, module in model.named_modules():
        if isinstance(module, ViewFeatureSelector):
            module.view_scale = num_cameras / trained_cameras
            scales[name] = module.view_scale
        elif isinstance(module, DeformableCrossAttention):
            module.camera_scale = trained_cameras / num_cameras
            scales[name] = module.camera_scale
    return scales
