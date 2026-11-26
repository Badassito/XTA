"""Native archive IO dispatch and safety for the interactive Slicer toolbar."""
import sys
from pathlib import Path
from types import SimpleNamespace

from tools.slicer import xta_toolbar_startup as startup


def test_reader_reuses_same_archive_and_preserves_other_active_review(tmp_path, monkeypatch):
    existing = tmp_path / "study.xtareview"
    existing.touch()
    other = tmp_path / "other.XTAREVIEW"
    other.touch()
    active = []
    messages = []

    def open_review(path):
        assert not active
        view = SimpleNamespace(store=SimpleNamespace(path=path), node=SimpleNamespace(GetID=lambda: "id"))
        active.append(view)
        return view

    monkeypatch.setitem(sys.modules, "__main__", SimpleNamespace(
        OpenXTAReview=open_review,
        _xta_current_native_window=lambda: active[-1] if active else None,
    ))
    monkeypatch.setitem(sys.modules, "vtk", SimpleNamespace(vtkCommand=SimpleNamespace(ErrorEvent=13)))
    parent = SimpleNamespace(loadedNodes=[], userMessages=lambda: SimpleNamespace(
        AddMessage=lambda event, message: messages.append((event, message)),
    ))
    reader = startup.XTANativeReviewIOFileReader(parent)
    assert reader.canLoadFile(str(other))
    assert not reader.canLoadFile(str(existing.with_suffix(".nrrd")))
    assert reader.load({"fileName": str(existing)})
    assert reader.load({"fileName": str(existing)})
    assert parent.loadedNodes == ["id"]
    assert len(active) == 1
    assert not reader.load({"fileName": str(other)})
    assert parent.loadedNodes == []
    assert len(active) == 1
    assert messages[0][0] == 13
    assert "CloseXTANativeReview()" in messages[0][1]


def test_reader_returns_both_full_field_working_nodes(tmp_path, monkeypatch):
    archive = tmp_path / "preview.xtareview"
    archive.touch()
    nodes = [SimpleNamespace(GetID=lambda: "layers"), SimpleNamespace(GetID=lambda: "global")]
    monkeypatch.setitem(sys.modules, "__main__", SimpleNamespace(
        OpenXTAReview=lambda path: SimpleNamespace(nodes=nodes), _xta_current_native_window=lambda: None,
    ))
    monkeypatch.setitem(sys.modules, "vtk", SimpleNamespace(vtkCommand=SimpleNamespace(ErrorEvent=13)))
    parent = SimpleNamespace(loadedNodes=[])
    reader = startup.XTANativeReviewIOFileReader(parent)
    assert reader.load({"fileName": str(archive)})
    assert parent.loadedNodes == ["layers", "global"]


def test_registered_reader_shim_is_private_and_install_is_idempotent(tmp_path, monkeypatch):
    register_calls = []
    load_calls = []
    existing_import = object()
    namespace = {"os": existing_import, "sys": existing_import, "callback": "old"}
    monkeypatch.setattr(startup.tempfile, "tempdir", str(tmp_path))

    def register(path):
        assert path.parent.parent == tmp_path
        assert "class XTANativeReviewIO" in path.read_text()
        namespace.pop("os")
        register_calls.append(path)

    def load(names):
        namespace.pop("sys")
        namespace["callback"] = "new"
        load_calls.append(names)
        return True

    manager = SimpleNamespace(registeredFileReaderCount=lambda name: len(load_calls))
    factory = SimpleNamespace(
        isRegistered=lambda name: bool(register_calls), loadModules=load, registerModule=register,
    )
    monkeypatch.setitem(sys.modules, "qt", SimpleNamespace(QFileInfo=Path))
    monkeypatch.setitem(sys.modules, "slicer", SimpleNamespace(app=SimpleNamespace(
        coreIOManager=lambda: manager, moduleManager=lambda: SimpleNamespace(factoryManager=lambda: factory),
    )))
    startup._register_file_reader(namespace)
    startup._register_file_reader(namespace)
    assert len(register_calls) == 1
    assert len(load_calls) == 1
    assert namespace["os"] is existing_import
    assert namespace["sys"] is existing_import
    assert namespace["callback"] == "new"
    assert register_calls[0].is_file()
    namespace["_xta_native_reader_module_dir"].cleanup()
