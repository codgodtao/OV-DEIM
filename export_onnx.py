"""Export the OV-DEIM open-vocabulary detection model to ONNX format.

Usage examples:

    # Export the LVIS / open-vocabulary variant (default config: base_l)
    python export_onnx.py --config base_l --checkpoint weights/ovdeim_l.pth --output ovdeim_l.onnx

    # Export the COCO variant
    python export_onnx.py --config coco_l --checkpoint weights/ovdeim_coco_l.pth --output ovdeim_coco_l.onnx

    # Export with a fixed batch size (recommended for TensorRT)
    python export_onnx.py --config base_s --checkpoint weights/ovdeim_s.pth --output ovdeim_s.onnx --batch-size 1 --dynamic-batch False

The exported ONNX model accepts two inputs:
    - ``image``      : float32[?, 3, 640, 640]  preprocessed image tensor (RGB, KeepRatioResize + LetterResize padded to 640x640, normalized to model expectations).
    - ``text_feats`` : float32[?, num_texts, text_dim]  text embeddings produced by MobileCLIP-B(LT) for the active vocabulary.

and returns three outputs:
    - ``scores`` : float32[?, num_top_queries]           sigmoid confidence scores (sorted descending).
    - ``labels`` : int64[?,  num_top_queries]            predicted class index into ``text_feats``.
    - ``boxes``  : float32[?, num_top_queries, 4]        boxes in ``cxcywh`` format, normalized to [0, 1]
                                                               relative to the 640x640 padded input.

See ``docs/ONNX_USAGE.md`` for the full post-processing recipe (de-normalising
the boxes back to the original image coordinate system).
"""
import argparse
import os
import sys
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import hydra
from hydra.core.config_store import ConfigStore

# Register all the project configs so that ``--config`` can resolve them.
from config.dinov3_ori.dinov3_l import DINOv3LConfig as base_l
from config.dinov3_ori.dinov3_m import DINOv3MConfig as base_m
from config.dinov3_ori.dinov3_s import DINOv3SConfig as base_s
from config.coco.dinov3_coco_l import DINOv3LConfig as coco_l
from config.coco.dinov3_coco_m import DINOv3MConfig as coco_m
from config.coco.dinov3_coco_s import DINOv3SConfig as coco_s

from model.backbone.dinov3_adapter import DINOv3STAs
from model.encoder.hybrid_encoder import HybridEncoder
from model.decoder.ovdeim_decoder import OVDEIMDecoder
from model.ovdeim import OVDEIM


_CONFIG_REGISTRY = {
    "base_l": base_l,
    "base_m": base_m,
    "base_s": base_s,
    "coco_l": coco_l,
    "coco_m": coco_m,
    "coco_s": coco_s,
}


class OVDEIMForOnnx(nn.Module):
    """Wrapper around :class:`OVDEIM` that is friendly to ``torch.onnx.export``.

    The wrapper:
      * replaces the dict-based ``targets`` argument with an explicit
        ``text_feats`` tensor input,
      * runs the detector in eval mode (``num_enc_queries = 0`` so that the
        decoder returns a single ``out`` dict),
      * applies the post-processing that does not depend on per-image
        geometry (sigmoid + top-k + gather). The remaining geometry
        restoration (un-pad / un-scale to the original image size) is left
        to the user, see ``docs/ONNX_USAGE.md``.
    """

    def __init__(self, model: OVDEIM, num_classes: int, num_top_queries: int = 300):
        super().__init__()
        # ``model`` already has SyncBatchNorm replaced by BatchNorm2d (see
        # ``convert_syncbn``) and is in eval mode.
        self.model = model
        self.num_classes = int(num_classes)
        self.num_top_queries = int(num_top_queries)

    def forward(self, image: torch.Tensor, text_feats: torch.Tensor):
        # Build the minimal ``targets`` dict expected by OVDEIM.forward.
        # In eval mode the decoder only reads ``text_feats``.
        targets = {"text_feats": text_feats}
        outputs = self.model(image, targets)

        # ``num_enc_queries`` is 0 in eval mode -> a plain dict is returned.
        if not isinstance(outputs, dict):
            raise RuntimeError(
                "OVDEIMForOnnx expects the decoder to return a single dict. "
                "Set decoder.num_enc_queries = 0 before exporting."
            )

        logits = outputs["pred_logits"]            # [B, Q, num_classes]
        boxes = outputs["pred_boxes"]              # [B, Q, 4]  cxcywh, normalized

        scores = torch.sigmoid(logits)            # [B, Q, num_classes]
        scores_flat = scores.flatten(1)          # [B, Q * num_classes]
        topk_scores, topk_index = torch.topk(
            scores_flat, self.num_top_queries, dim=-1
        )
        labels = topk_index % self.num_classes
        box_index = topk_index // self.num_classes
        # Gather the matching boxes: [B, K, 4]
        gather_index = box_index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1])
        topk_boxes = boxes.gather(dim=1, index=gather_index)

        return topk_scores, labels.to(torch.int64), topk_boxes


def convert_syncbn(model: nn.Module) -> nn.Module:
    """Replace ``SyncBatchNorm`` layers with regular ``BatchNorm2d``.

    ``SyncBatchNorm`` requires a distributed process group and is not friendly
    to single-process ONNX tracing. In eval mode both layers compute the same
    forward pass from the running statistics, so we can safely swap the module
    type while preserving all learned parameters and buffers.
    """

    def _swap(module: nn.Module) -> nn.Module:
        for name, child in module.named_children():
            if isinstance(child, nn.SyncBatchNorm):
                bn = nn.BatchNorm2d(
                    child.num_features,
                    eps=child.eps,
                    momentum=child.momentum,
                    affine=child.affine,
                    track_running_stats=child.track_running_stats,
                )
                if child.affine:
                    bn.weight.data.copy_(child.weight.data)
                    bn.bias.data.copy_(child.bias.data)
                if child.track_running_stats:
                    bn.running_mean.data.copy_(child.running_mean.data)
                    bn.running_var.data.copy_(child.running_var.data)
                    bn.num_batches_tracked.data.copy_(child.num_batches_tracked.data)
                bn.eval()
                setattr(module, name, bn)
            else:
                _swap(child)
        return module

    return _swap(model)


def load_checkpoint(model: OVDEIM, checkpoint_path: str) -> None:
    """Load weights from a checkpoint file (supports several common formats)."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "ema_state_dict" in checkpoint:
        print("  -> loading ema_state_dict")
        state_dict = checkpoint["ema_state_dict"]
        if isinstance(state_dict, dict) and "module" in state_dict:
            state_dict = state_dict["module"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        print("  -> loading model_state_dict")
        state_dict = checkpoint["model_state_dict"]
    else:
        print("  -> loading raw state_dict")
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        # Skip denoising weights (not used in eval mode).
        if "decoder.denoising_class_embed.weight" in k:
            continue
        # Strip optional ``module.`` prefix from DDP checkpoints.
        clean_k = k[7:] if k.startswith("module.") else k
        new_state_dict[clean_k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"  -> missing keys ({len(missing)}): {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  -> unexpected keys ({len(unexpected)}): {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")


def build_model(args) -> OVDEIM:
    """Build the OVDEIM model from a hydra config name."""
    config_name = args.config
    if config_name not in _CONFIG_REGISTRY:
        raise ValueError(
            f"Unknown config '{config_name}'. Available: {list(_CONFIG_REGISTRY.keys())}"
        )

    cs = ConfigStore.instance()
    for name, node in _CONFIG_REGISTRY.items():
        cs.store(name=name, node=node)

    # Initialise hydra only once (the project scripts do it globally).
    try:
        hydra.initialize(config_path="config", version_base=None)
    except ValueError:
        # Already initialised in this process.
        pass

    cfg = hydra.compose(config_name=config_name)

    backbone = DINOv3STAs(**cfg.backbone)
    encoder = HybridEncoder(**cfg.encoder)
    decoder = OVDEIMDecoder(**cfg.decoder)
    model = OVDEIM(backbone, encoder, decoder, **cfg.model)

    # Disable the extra encoder queries path: eval inference returns a single dict.
    decoder.num_enc_queries = 0
    return model, cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export OV-DEIM detection model to ONNX format."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="base_l",
        choices=list(_CONFIG_REGISTRY.keys()),
        help="Hydra config name (e.g. base_l, coco_s).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the .pth checkpoint (EMA or plain state dict).",
    )
    adda = parser.add_argument
    adda("--output", type=str, default="ovdeim.onnx", help="Output ONNX file path.")
    adda("--opset", type=int, default=17, help="ONNX opset version.")
    adda("--num-top-queries", type=int, default=300, help="Number of top queries to keep.")
    adda("--num-texts", type=int, default=None,
         help="Number of text embeddings per sample. Defaults to data.num_training_classes.")
    adda("--batch-size", type=int, default=1, help="Static batch size when --dynamic-batch is False.")
    adda("--dynamic-batch", action="store_true", default=True,
         help="Allow dynamic batch dimension (default). Use --no-dynamic-batch to disable.")
    parser.add_argument("--no-dynamic-batch", dest="dynamic_batch", action="store_false")
    adda("--simplify", action="store_true", default=True,
         help="Run onnx-simplifier if installed (default True). Use --no-simplify to disable.")
    parser.add_argument("--no-simplify", dest="simplify", action="store_false")
    adda("--verify", action="store_true", default=True,
         help="Compare ONNX outputs against PyTorch outputs (default True).")
    parser.add_argument("--no-verify", dest="verify", action="store_false")
    return parser.parse_args()


def maybe_simplify(onnx_path: str) -> bool:
    """Try to simplify the ONNX model with onnx-simplifier."""
    try:
        import onnxsim  # type: ignore
    except ImportError:
        print("onnx-simplifier not installed, skipping simplification.")
        return False
    import onnx

    print("Simplifying ONNX model ...")
    model = onnx.load(onnx_path)
    model_sim, check = onnxsim.simplify(model)
    if check:
        onnx.save(model_sim, onnx_path)
        print("  -> simplification succeeded.")
        return True
    print("  -> simplifier check failed, keeping the original model.")
    return False


def verify_onnx(onnx_path: str, wrapper: OVDEIMForOnnx, image: torch.Tensor, text_feats: torch.Tensor):
    """Compare ONNX runtime outputs with PyTorch outputs."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed, skipping verification.")
        return

    print("Verifying ONNX outputs against PyTorch ...")
    wrapper.eval()
    with torch.no_grad():
        pt_scores, pt_labels, pt_boxes = wrapper(image, text_feats)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_inputs = {
        "image": image.cpu().numpy(),
        "text_feats": text_feats.cpu().numpy(),
    }
    onnx_outputs = sess.run(None, onnx_inputs)

    for name, pt_out, onnx_out in zip(
        ["scores", "labels", "boxes"],
        [pt_scores, pt_labels, pt_boxes],
        onnx_outputs,
    ):
        max_diff = (pt_out.float() - torch.from_numpy(onnx_out).float()).abs().max().item()
        status = "OK" if max_diff < 1e-3 else "WARN"
        print(f"  -> {name}: max abs diff = {max_diff:.6e}  [{status}]")


def main():
    args = parse_args()

    # Build model + config.
    model, cfg = build_model(args)
    load_checkpoint(model, args.checkpoint)

    # Convert SyncBatchNorm -> BatchNorm2d, then switch to eval mode.
    model = convert_syncbn(model)
    model.eval()

    num_classes = cfg.decoder.num_classes
    wrapper = OVDEIMForOnnx(model, num_classes=num_classes, num_top_queries=args.num_top_queries)
    wrapper.eval()

    num_texts = args.num_texts if args.num_texts is not None else cfg.data.num_training_classes
    text_dim = cfg.model.text_dim
    img_size = cfg.data.img_scale  # (H, W), typically (640, 640)
    h, w = int(img_size[0]), int(img_size[1])

    batch = args.batch_size
    dummy_image = torch.randn(batch, 3, h, w, dtype=torch.float32)
    dummy_text = torch.randn(batch, num_texts, text_dim, dtype=torch.float32)

    print(f"Exporting to ONNX: {args.output}")
    print(f"  image shape      : {tuple(dummy_image.shape)}")
    print(f"  text_feats shape : {tuple(dummy_text.shape)}")
    print(f"  num_classes      : {num_classes}")
    print(f"  num_top_queries  : {args.num_top_queries}")
    print(f"  opset            : {args.opset}")
    print(f"  dynamic batch    : {args.dynamic_batch}")

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            "image": {0: "batch"},
            "text_feats": {0: "batch"},
            "scores": {0: "batch"},
            "labels": {0: "batch"},
            "boxes": {0: "batch"},
        }

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_image, dummy_text),
            args.output,
            input_names=["image", "text_feats"],
            output_names=["scores", "labels", "boxes"],
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=True,
        )
    print(f"ONNX export finished -> {args.output}")

    if args.simplify:
        maybe_simplify(args.output)

    if args.verify:
        verify_onnx(args.output, wrapper, dummy_image, dummy_text)

    print("Done.")


if __name__ == "__main__":
    main()
