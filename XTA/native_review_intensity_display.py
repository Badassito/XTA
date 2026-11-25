"""Compose an aligned navigation overview and native grayscale window in Slicer."""

from dataclasses import dataclass


@dataclass
class _Foreground:
    previous_id: str | None
    opacity: float
    compositing: int
    assigned_id: str | None = None


class NativeIntensityDisplay:
    """Borrow Slicer's foreground layer for the exact native window."""

    def __init__(self, scene, modified_event):
        self._scene = scene
        self._modified_event = modified_event
        self._foregrounds = {}
        self._displays = ()
        self._observers = []
        self._adjusting = False

    def _configure_contrast(self, overview, window):
        for volume in (overview, window):
            if volume.GetDisplayNode() is None:
                volume.CreateDefaultDisplayNodes()
        displays = (overview.GetDisplayNode(), window.GetDisplayNode())
        if displays == self._displays:
            return
        self._remove_observers()
        source = displays[0]
        contrast = (source.GetWindow(), source.GetLevel()) if self._displays else (255.0, 127.5)
        for display in displays:
            display.SetAutoWindowLevel(False)
            display.SetApplyThreshold(False)
            display.SetWindowLevel(*contrast)
            self._observers.append((display, display.AddObserver(self._modified_event, self._sync_contrast)))
        self._displays = displays

    def _sync_contrast(self, caller, _event):
        if self._adjusting or caller not in self._displays:
            return
        contrast = caller.GetWindow(), caller.GetLevel()
        self._adjusting = True
        try:
            for other in self._displays:
                if other != caller and (other.GetWindow(), other.GetLevel()) != contrast:
                    other.SetWindowLevel(*contrast)
        finally:
            self._adjusting = False

    def show(self, overview, window, *, overview_only=False):
        """Keep overview context outside the native reslice's valid stencil."""
        self._configure_contrast(overview, window)
        foreground = None if overview_only else window.GetID()
        for i in range(self._scene.GetNumberOfNodesByClass("vtkMRMLSliceCompositeNode")):
            composite = self._scene.GetNthNodeByClass(i, "vtkMRMLSliceCompositeNode")
            node_id = composite.GetID()
            if node_id not in self._foregrounds:
                self._foregrounds[node_id] = _Foreground(
                    composite.GetForegroundVolumeID(), composite.GetForegroundOpacity(),
                    composite.GetCompositing(),
                )
            self._foregrounds[node_id].assigned_id = foreground
            modifying = composite.StartModify()
            try:
                composite.SetForegroundVolumeID(foreground)
                composite.SetForegroundOpacity(1.0)
                composite.SetCompositing(composite.Alpha)
            finally:
                composite.EndModify(modifying)

    def _remove_observers(self):
        for node, tag in self._observers:
            node.RemoveObserver(tag)
        self._observers.clear()

    def close(self, *, restore):
        self._remove_observers()
        self._displays = ()
        if restore:
            for node_id, saved in self._foregrounds.items():
                composite = self._scene.GetNodeByID(node_id)
                if composite is None or composite.GetForegroundVolumeID() != saved.assigned_id:
                    continue
                modifying = composite.StartModify()
                try:
                    previous = saved.previous_id
                    if previous is not None and self._scene.GetNodeByID(previous) is None:
                        previous = None
                    composite.SetForegroundVolumeID(previous)
                    composite.SetForegroundOpacity(saved.opacity)
                    composite.SetCompositing(saved.compositing)
                finally:
                    composite.EndModify(modifying)
        self._foregrounds.clear()
