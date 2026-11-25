"""Start the portable XTA toolbar in the interactive Slicer namespace."""
import logging
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from zipfile import ZipFile


_FILE_TYPE = "XTANativeReviewFile"
_MODULE_NAME = "XTANativeReviewIO"


class XTANativeReviewIOFileReader:
    """Expose XTA archives to Slicer's existing Add Data file IO manager."""

    def __init__(self, parent):
        self.parent = parent

    def description(self):
        return "XTA Native Review"

    def fileType(self):
        return _FILE_TYPE

    def extensions(self):
        return ["XTA Native Review (*.xtareview)"]

    def canLoadFile(self, path):
        return Path(os.fsdecode(path)).suffix.lower() == ".xtareview"

    def load(self, properties):
        import __main__
        import vtk

        self.parent.loadedNodes = []
        try:
            path = Path(os.fsdecode(properties["fileName"])).resolve(strict=True)
            if not self.canLoadFile(path):
                raise ValueError("Choose an .xtareview archive.")
            review = __main__._xta_current_native_window()
            if review is None:
                review = __main__.OpenXTANativeReview(path)
            elif path != review.store.path:
                raise RuntimeError("Close the current native review before opening another. "
                                   "Use CloseXTANativeReview() in the Slicer Python console.")
            self.parent.loadedNodes = [review.node.GetID()]
            return True
        except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
            self.parent.userMessages().AddMessage(vtk.vtkCommand.ErrorEvent, str(error))
            return False


def _register_file_reader(namespace):
    import qt
    import slicer

    io = slicer.app.coreIOManager()
    if io.registeredFileReaderCount(_FILE_TYPE):
        return
    factory = slicer.app.moduleManager().factoryManager()
    staging = namespace.get("_xta_native_reader_module_dir")
    # The Slicer 5.12 module factory uses temporary names in the console
    # namespace and may delete pre-existing imports such as os and sys.
    namespace.setdefault("os", os)
    namespace.setdefault("sys", sys)
    existing_console_names = dict(namespace)
    try:
        if factory.isRegistered(_MODULE_NAME):
            if staging is None:
                raise RuntimeError("Another Slicer module already uses the XTA native reader name.")
        else:
            # Slicer 5.12 registers the module's FileReader in its C++ IO
            # manager when the module sets up.
            staging = tempfile.TemporaryDirectory(prefix="xta-slicer-io-")
            path = Path(staging.name) / (_MODULE_NAME + ".py")
            path.write_text(
                "from slicer.ScriptedLoadableModule import ScriptedLoadableModule\n"
                "from xta_toolbar_startup import XTANativeReviewIOFileReader\n"
                "\n"
                "class XTANativeReviewIO(ScriptedLoadableModule):\n"
                "    def __init__(self, parent):\n"
                "        super().__init__(parent)\n"
                "        parent.title = 'XTA Native Review IO'\n"
                "        parent.hidden = True\n",
                encoding="utf-8",
            )
            factory.registerModule(qt.QFileInfo(str(path)))
            if not factory.isRegistered(_MODULE_NAME):
                staging.cleanup()
                raise RuntimeError("Slicer could not register the scripted file reader module.")
            namespace["_xta_native_reader_module_dir"] = staging
        namespace.setdefault("os", existing_console_names["os"])
        namespace.setdefault("sys", existing_console_names["sys"])
        loaded = factory.loadModules([_MODULE_NAME])
    finally:
        for name, value in existing_console_names.items():
            namespace.setdefault(name, value)
    if not loaded or not io.registeredFileReaderCount(_FILE_TYPE):
        raise RuntimeError("Slicer could not activate the native review file reader.")


def install():
    import __main__

    archive = getattr(__loader__, "archive", None)
    if archive is None:
        script = Path(__file__).with_name("xta_slicerrc.py")
        data = script.read_bytes()
    else:
        with ZipFile(archive) as bundle:
            data = bundle.read("xta_slicerrc.py")
        script = Path(archive) / "xta_slicerrc.py"
    exec(compile(data, str(script), "exec"), vars(__main__))
    try:
        _register_file_reader(vars(__main__))
    except (AttributeError, RuntimeError, OSError) as error:
        logging.warning("XTA Add Data reader unavailable. Open .xtareview in Importer. %s", error)
