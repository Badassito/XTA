"""Export a binary YOLO semantic checkpoint with its raw class logits intact.

Ultralytics normally reduces semantic ONNX and OpenVINO exports to a class map.
That removes the confidence needed by XTA. This exporter traces the model's
ordinary evaluation path instead, producing [batch, 1 or 2, height/8, width/8]
floating-point logits. XTA upsamples and decodes those logits at inference time.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Semantic YOLO .pt checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Destination .onnx or OpenVINO .xml")
    parser.add_argument("--imgsz", type=int, default=640, help="Square input size (multiple of 32)")
    parser.add_argument("--batch", type=int, default=1, help="Static export batch size")
    parser.add_argument(
        "--compress-fp16", action="store_true",
        help="Compress OpenVINO weights to FP16 (the default preserves FP32)",
    )
    args = parser.parse_args(argv)
    if not args.model.is_file() or args.model.suffix.lower() != ".pt":
        parser.error("--model must be an existing .pt checkpoint")
    if args.output.suffix.lower() not in {".onnx", ".xml"}:
        parser.error("--output must end in .onnx or .xml")
    if args.output.resolve() == args.model.resolve():
        parser.error("--output must differ from --model")
    if args.output.exists() or (args.output.suffix.lower() == ".xml" and args.output.with_suffix(".bin").exists()):
        parser.error("Refusing to replace an existing export")
    if args.imgsz < 32 or args.imgsz % 32:
        parser.error("--imgsz must be a positive multiple of 32")
    if args.batch < 1:
        parser.error("--batch must be positive")
    if args.compress_fp16 and args.output.suffix.lower() != ".xml":
        parser.error("--compress-fp16 applies only to OpenVINO .xml exports")
    return args


def prepare_model(path: Path):
    """Load a semantic model on CPU and retain the evaluation logits path."""
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.modules import SemanticSegment

    wrapper = YOLO(str(path.resolve()), task="semantic")
    if wrapper.task != "semantic":
        raise ValueError(f"Expected a YOLO semantic checkpoint, got task={wrapper.task!r}")
    model = wrapper.model.eval().float().cpu()
    head = model.model[-1]
    if not isinstance(head, SemanticSegment):
        raise ValueError(f"Expected a SemanticSegment head, got {type(head).__name__}")
    if int(head.nc) not in (1, 2):
        raise ValueError(f"XTA requires binary semantic logits (1 or 2 channels); model has {head.nc}")
    channels = next((module.in_channels for module in model.modules() if isinstance(module, torch.nn.Conv2d)), None)
    if channels is None or channels < 1:
        raise ValueError("Cannot determine the checkpoint's input channel count")
    head.export = False  # Ultralytics' export path bakes argmax for ONNX and OpenVINO.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, int(channels), int(head.nc)


def export_model(model, example, output: Path, *, compress_fp16: bool = False) -> tuple[int, ...]:
    """Validate the raw output, then write one static ONNX or OpenVINO graph."""
    import torch

    model.eval().cpu()
    with torch.inference_mode():
        logits = model(example)
        if (not isinstance(logits, torch.Tensor) or logits.ndim != 4
                or not logits.dtype.is_floating_point or logits.shape[0] != example.shape[0]
                or logits.shape[1] not in (1, 2)):
            raise ValueError("Expected one floating-point [batch, 1 or 2, height, width] logits tensor")
        shape = tuple(int(value) for value in logits.shape)
        if output.suffix.lower() == ".onnx":
            torch.onnx.export(
                model, example, str(output), input_names=["images"], output_names=["logits"],
                opset_version=17, do_constant_folding=True, dynamo=False,
            )
        elif output.suffix.lower() == ".xml":
            import openvino as ov

            converted = ov.convert_model(model, example_input=example)
            ov.save_model(converted, str(output), compress_to_fp16=compress_fp16)
        else:
            raise ValueError("Output must end in .onnx or .xml")
    return shape


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Ultralytics initializes settings at import time; keep them in a writable
    # task location even on machines where the roaming profile is locked down.
    os.environ.setdefault("YOLO_CONFIG_DIR", str(output.parent / "ultralytics-config"))
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")

    import torch

    model, channels, classes = prepare_model(args.model)
    example = torch.zeros((args.batch, channels, args.imgsz, args.imgsz), dtype=torch.float32)
    shape = export_model(model, example, output, compress_fp16=args.compress_fp16)
    print(f"Exported {output} with input {tuple(example.shape)} and raw logits {shape} ({classes} class channel(s))")


if __name__ == "__main__":
    main()
