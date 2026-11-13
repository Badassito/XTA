"""Run the unified mode-aware CLI with ``python -m XTA``."""

from .cli import run


if __name__ == "__main__":
    import os

    if os.name == "nt" and "XTA_YOLO_SWEEP_CUTOFF_HUNDREDTHS" in os.environ:
        # The coordinator targets this child's console process group with
        # CTRL_BREAK_EVENT so the normal TTA teardown can drain its workers.
        import signal
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    run()
