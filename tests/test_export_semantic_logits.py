"""CPU checks for semantic exports that retain usable confidence logits."""
from pathlib import Path

import numpy as np
import pytest

from tools.export_semantic_logits import export_model, parse_args


def test_parser_rejects_class_map_target_and_invalid_size(tmp_path: Path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    with pytest.raises(SystemExit):
        parse_args(["--model", str(checkpoint), "--output", str(tmp_path / "map.png")])
    with pytest.raises(SystemExit):
        parse_args(["--model", str(checkpoint), "--output", str(tmp_path / "logits.onnx"), "--imgsz", "33"])


@pytest.mark.parametrize("suffix", [".onnx", ".xml"])
def test_export_retains_float_logits_and_runtime_values(tmp_path: Path, suffix: str):
    torch = pytest.importorskip("torch")
    ov = pytest.importorskip("openvino")
    if suffix == ".onnx":
        onnx = pytest.importorskip("onnx")

    class TinySemantic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.Conv2d(1, 1, 3, stride=2, padding=1, bias=True)
            torch.nn.init.constant_(self.classifier.weight, 0.125)
            torch.nn.init.constant_(self.classifier.bias, -0.25)

        def forward(self, image):
            return self.classifier(image)

    model = TinySemantic().eval()
    example = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8) / 64
    output = tmp_path / f"logits{suffix}"
    shape = export_model(model, example, output)
    assert shape == (1, 1, 4, 4)
    if suffix == ".onnx":
        graph = onnx.load(str(output))
        onnx.checker.check_model(graph)
        assert len(graph.graph.output) == 1
        assert graph.graph.output[0].name == "logits"
    else:
        assert output.with_suffix(".bin").is_file()

    compiled = ov.Core().compile_model(str(output), "CPU")
    actual = compiled({0: example.numpy()})[compiled.output(0)]
    expected = model(example).detach().numpy()
    assert actual.shape == expected.shape
    assert np.issubdtype(actual.dtype, np.floating)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
