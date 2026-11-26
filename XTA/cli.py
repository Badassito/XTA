"""Dependency-light mode dispatcher for the unified TTA/PTA/LTA launcher."""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

from .unification.context import activate_unified_launch
from . import __version__


SCRIPT_VERSION = __version__
SCRIPT_BASENAME = f"GPT-6-Astra-Ultra_v{SCRIPT_VERSION}_SLURM.py"
MODE_CHOICES = ("tta", "pta", "lta")


def build_argparser(*, dispatch_only: bool = False) -> argparse.ArgumentParser:
    """Build the dependency-free compatibility-launcher parser.

    The dispatch-only form deliberately omits ``-h/--help`` so help following a
    selected mode is handled by that mode's complete parser.
    """

    parser = argparse.ArgumentParser(
        prog=SCRIPT_BASENAME,
        usage=f"{SCRIPT_BASENAME} --mode {{tta,pta,lta}} ...",
        epilog="For paired-volume registration and reslicing, use 'xta register --help'.",
        description=(
            "Mode-aware launcher for volume test-time, pretraining, and label-time "
            "augmentation. Select a mode, then supply only that mode's flags. Use "
            "'--mode tta --help', '--mode pta --help', or '--mode lta --help' "
            "for mode-specific help."
        ),
        allow_abbrev=False,
        add_help=not bool(dispatch_only),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {SCRIPT_VERSION}",
    )
    parser.add_argument("--mode", required=True, choices=MODE_CHOICES)
    return parser


def _has_mode_argument(arguments: Sequence[str]) -> bool:
    return any(
        str(token) == "--mode" or str(token).startswith("--mode=")
        for token in arguments
    )


def _mode_argument_count(arguments: Sequence[str]) -> int:
    return sum(
        1
        for token in arguments
        if str(token) == "--mode" or str(token).startswith("--mode=")
    )


def _option_argument_count(arguments: Sequence[str], option: str) -> int:
    """Count exact scalar-option occurrences, including ``--flag=value`` form."""

    option_name = str(option)
    return sum(
        1
        for token in arguments
        if str(token) == option_name or str(token).startswith(f"{option_name}=")
    )


@contextlib.contextmanager
def _mode_sys_argv(arguments: Sequence[str]) -> Iterator[None]:
    """Expose mode-local arguments to the established TTA entry point."""

    previous = sys.argv
    sys.argv = [str(previous[0]), *(str(value) for value in arguments)]
    try:
        yield
    finally:
        sys.argv = previous


def _run_tta(arguments: Sequence[str]) -> None:
    """Validate with the existing TTA grammar, then run its production pipeline."""

    mode_arguments = [str(value) for value in arguments]
    # Validate before importing the production pipeline. This preserves argparse's
    # unavailable/foreign-flag errors on dependency-light hosts and keeps the TTA
    # defaults and accepted flags authoritative.
    from .config import build_argparser as build_tta_argparser

    parser = build_tta_argparser()
    if _option_argument_count(mode_arguments, "--channel_format") > 1:
        parser.error("--channel_format may be provided only once")
    parsed = parser.parse_args(mode_arguments)
    from .config import (
        parse_confidence_sweep_options,
        resolve_auto_positive_int,
        resolve_backend_batches,
        resolve_backend_devices,
        resolve_backend_models,
        resolve_backend_precisions,
        resolve_cartesian_views,
        resolve_channel_format,
        resolve_azimuthal_view_requests,
        resolve_interpolation_settings,
        resolve_postprocessing_options,
        resolve_radial_view_requests,
        resolve_save_request,
        resolve_spherical_view_requests,
        resolve_tilted_view_groups,
        resolve_tta_angles,
        activate_sam_crop_mode,
        activate_sam_adaptive_crop,
    )
    from .tta_augmentation_config import resolve_tta_augmentation
    from .reconciliation_policy import resolve_reconciliation
    from .scheduler_preflight import preflight_tta_scheduler
    try:
        parse_confidence_sweep_options(parser, parsed, mode_arguments)
        save_request = resolve_save_request(parsed.save)
        resolve_postprocessing_options(parsed.postprocessing)
        resolve_channel_format(parsed.channel_format)
        resolve_tta_angles(parsed.angle)
        resolve_cartesian_views(parsed.enable_cartesian)
        resolve_tilted_view_groups(parsed.enable_tilted)
        azimuthal = resolve_azimuthal_view_requests(parsed.enable_azimuthal)
        resolve_radial_view_requests(parsed.enable_radial)
        spherical = resolve_spherical_view_requests(parsed.enable_spherical)
        if spherical and parsed.imgsz <= 0:
            raise ValueError('Spherical QSC patches require --imgsz > 0')
        devices = resolve_backend_devices(parsed.device)
        if azimuthal and devices.cpu and not devices.gpu_devices:
            raise ValueError('OpenVINO cannot execute the requested Azimuthal detector views')
        models = resolve_backend_models(parsed.model)
        resolve_backend_precisions(parsed.quantize, devices)
        resolve_backend_batches(parsed.batch, devices)
        for field in ('cpu_instances', 'cpu_threads', 'cpu_streams', 'cpu_infer_requests'):
            resolve_auto_positive_int(getattr(parsed, field), flag_name=f'--{field}')
        interpolation = resolve_interpolation_settings(parsed, models, devices)
        resolve_tta_augmentation(parsed, gpu_devices=devices.gpu_devices, cpu_enabled=devices.cpu)
        resolve_reconciliation(parsed)
        preflight_tta_scheduler(
            gpu_device_count=len(devices.gpu_devices), cpu_enabled=devices.cpu,
            tile_configs=parsed.enable_tile or (),
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    if parsed.conf_sweep is not None:
        from .confidence_sweep import run_confidence_sweep

        output = Path(parsed.output).expanduser() if parsed.output else Path.cwd() / Path(parsed.input).expanduser().resolve().stem
        run_confidence_sweep(
            mode_arguments,
            cutoffs_hundredths=parsed.conf_sweep.levels_hundredths,
            output_root=output,
            temp_root=Path(parsed.temp).expanduser() if parsed.temp else None,
            min_conf=parsed.min_conf,
            save_request=save_request,
        )
        return

    # The established entry point and pipeline consume sys.argv. Remove the unified mode
    # selector for the duration of the call rather than teaching the TTA parser a new flag.
    from .tta_mode import run as run_tta

    with activate_unified_launch(
        version=SCRIPT_VERSION,
        launcher=SCRIPT_BASENAME,
        mode="tta",
        mode_arguments=mode_arguments,
    ):
        with (activate_sam_crop_mode(interpolation.sam_crop_mode),
              activate_sam_adaptive_crop(interpolation.sam_adaptive_crop)):
            with _mode_sys_argv(mode_arguments):
                run_tta()


def _run_pta(arguments: Sequence[str]) -> None:
    """Load and invoke the PTA implementation only after PTA mode is selected."""

    from .pta_mode import run as run_pta

    mode_arguments = [str(value) for value in arguments]
    with activate_unified_launch(
        version=SCRIPT_VERSION,
        launcher=SCRIPT_BASENAME,
        mode="pta",
        mode_arguments=mode_arguments,
    ):
        run_pta(mode_arguments)


def _run_lta(arguments: Sequence[str]) -> None:
    """Load and invoke the LTA boundary only after LTA mode is selected."""

    from .lta_mode import run as run_lta

    mode_arguments = [str(value) for value in arguments]
    with activate_unified_launch(
        version=SCRIPT_VERSION,
        launcher=SCRIPT_BASENAME,
        mode="lta",
        mode_arguments=mode_arguments,
    ):
        run_lta(mode_arguments)


def run(argv: Sequence[str] | None = None) -> None:
    """Dispatch one compatibility-launcher invocation.

    ``argv`` excludes the program name. Passing it explicitly keeps dispatch tests
    independent from process-global command-line state.
    """

    arguments = [str(value) for value in (sys.argv[1:] if argv is None else argv)]
    if arguments[:1] == ["register"]:
        from .registration.cli import main as run_registration

        run_registration(arguments[1:])
        return

    # Top-level discovery must not import OpenCV, SciPy, model runtimes, or any
    # mode implementation. Once a mode is present, its own --help remains authoritative.
    if not arguments or (
        not _has_mode_argument(arguments)
        and any(token in {"-h", "--help"} for token in arguments)
    ):
        build_argparser().parse_args(arguments)
        return

    dispatch_parser = build_argparser(dispatch_only=True)
    namespace, mode_arguments = dispatch_parser.parse_known_args(arguments)
    if _mode_argument_count(arguments) != 1:
        dispatch_parser.error("--mode must be specified exactly once")

    if namespace.mode == "tta":
        _run_tta(mode_arguments)
        return
    if namespace.mode == "pta":
        _run_pta(mode_arguments)
        return
    if namespace.mode == "lta":
        _run_lta(mode_arguments)
        return
    raise AssertionError(f"unhandled mode {namespace.mode!r}")


__all__ = [
    "MODE_CHOICES",
    "SCRIPT_BASENAME",
    "SCRIPT_VERSION",
    "build_argparser",
    "run",
]
