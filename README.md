# XTA

XTA processes large image volumes in three modes:

- **TTA** combines model predictions from transformed views of a volume.
- **PTA** generates pretraining images and labels from labeled volumes.
- **LTA** propagates authoritative annotations with a local SAM model.

Use the mode-specific command help for the current interface:

```text
python -m XTA --mode tta --help
python -m XTA --mode pta --help
python -m XTA --mode lta --help
```

The versioned launcher and the installed `xta` command use the same dispatcher.
Runtime dependencies and optional backends are declared in [pyproject.toml](pyproject.toml).

## Guides

- [v25.0.0 options and migration](docs/topology_and_cpr.md)
- [SAM interpolation and extrapolation](docs/sam_interpolation.md)
- [Dynamic LTA crops](docs/lta_dynamic_crops.md)
- [PTA centerline cross-sections](docs/pta_cpr.md)
- [Slicer cleanup workflow](docs/slicer_pta_cleanup.md)
- [Projection coverage contracts](docs/projection_coverage.md)
- [Moving completed runs](docs/run_transport.md)

## Development

Start with the [maintenance workflow](docs/maintenance.md) for source inventory,
packaging, validation, generated artifacts, and release checks. The
[tools index](tools/README.md) distinguishes user utilities from source-checkout
qualification tools. [ARCHITECTURE.md](ARCHITECTURE.md) describes runtime ownership
and numerical contracts; [AGENTS.md](AGENTS.md) governs repository-specific work.
