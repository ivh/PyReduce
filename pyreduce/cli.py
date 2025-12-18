"""Click-based CLI for PyReduce.

Usage:
    uv run reduce --help
    uv run reduce bias INSTRUMENT --files bias/*.fits --output output/
    uv run reduce run reduction.yaml
"""

from __future__ import annotations

from glob import glob
from pathlib import Path

import click

from .instruments.instrument_info import load_instrument
from .pipeline import Pipeline


@click.group()
@click.version_option(package_name="pyreduce-astro")
def cli():
    """PyReduce echelle spectrograph reduction pipeline."""
    pass


@cli.command()
@click.argument("instrument")
@click.option(
    "--files",
    "-f",
    multiple=True,
    help="Input FITS files (can use glob patterns)",
)
@click.option("--output", "-o", default=".", help="Output directory")
@click.option("--mode", "-m", default="", help="Instrument mode (e.g., RED, BLUE)")
def bias(instrument: str, files: tuple[str, ...], output: str, mode: str):
    """Create master bias from bias frames."""
    inst = load_instrument(instrument)
    file_list = _expand_globs(files)

    if not file_list:
        raise click.ClickException("No input files specified. Use --files option.")

    click.echo(f"Creating master bias from {len(file_list)} files...")
    Pipeline(inst, output, mode=mode).bias(file_list).run()
    click.echo(f"Master bias saved to {output}")


@cli.command()
@click.argument("instrument")
@click.option(
    "--files",
    "-f",
    multiple=True,
    help="Input FITS files (can use glob patterns)",
)
@click.option("--output", "-o", default=".", help="Output directory")
@click.option("--mode", "-m", default="", help="Instrument mode")
def flat(instrument: str, files: tuple[str, ...], output: str, mode: str):
    """Create master flat from flat frames."""
    inst = load_instrument(instrument)
    file_list = _expand_globs(files)

    if not file_list:
        raise click.ClickException("No input files specified. Use --files option.")

    click.echo(f"Creating master flat from {len(file_list)} files...")
    Pipeline(inst, output, mode=mode).flat(file_list).run()
    click.echo(f"Master flat saved to {output}")


@cli.command()
@click.argument("instrument")
@click.option(
    "--files",
    "-f",
    multiple=True,
    help="Flat files for tracing (optional if flat already exists)",
)
@click.option("--output", "-o", default=".", help="Output directory")
@click.option("--mode", "-m", default="", help="Instrument mode")
def trace(instrument: str, files: tuple[str, ...], output: str, mode: str):
    """Trace echelle orders on flat field."""
    inst = load_instrument(instrument)
    file_list = _expand_globs(files) if files else None

    click.echo("Tracing echelle orders...")
    pipe = Pipeline(inst, output, mode=mode)
    if file_list:
        pipe = pipe.flat(file_list)
    pipe.trace_orders(file_list).run()
    click.echo(f"Order trace saved to {output}")


@cli.command()
@click.argument("instrument")
@click.option(
    "--files",
    "-f",
    multiple=True,
    required=True,
    help="Wavelength calibration files (ThAr, etc.)",
)
@click.option("--output", "-o", default=".", help="Output directory")
@click.option("--mode", "-m", default="", help="Instrument mode")
def wavecal(instrument: str, files: tuple[str, ...], output: str, mode: str):
    """Perform wavelength calibration."""
    inst = load_instrument(instrument)
    file_list = _expand_globs(files)

    if not file_list:
        raise click.ClickException("No input files specified. Use --files option.")

    click.echo(f"Running wavelength calibration with {len(file_list)} files...")
    Pipeline(inst, output, mode=mode).wavelength_calibration(file_list).run()
    click.echo(f"Wavelength calibration saved to {output}")


@cli.command()
@click.argument("instrument")
@click.option(
    "--files",
    "-f",
    multiple=True,
    required=True,
    help="Science observation files",
)
@click.option("--output", "-o", default=".", help="Output directory")
@click.option("--mode", "-m", default="", help="Instrument mode")
@click.option("--target", "-t", default="", help="Target name")
def extract(
    instrument: str, files: tuple[str, ...], output: str, mode: str, target: str
):
    """Extract spectra from science frames."""
    inst = load_instrument(instrument)
    file_list = _expand_globs(files)

    if not file_list:
        raise click.ClickException("No input files specified. Use --files option.")

    click.echo(f"Extracting spectra from {len(file_list)} files...")
    Pipeline(inst, output, mode=mode, target=target).extract(file_list).run()
    click.echo(f"Extracted spectra saved to {output}")


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option(
    "--steps",
    "-s",
    default="all",
    help="Steps to run (comma-separated, or 'all')",
)
@click.option("--skip-existing", is_flag=True, help="Skip steps with existing output")
def run(config_file: str, steps: str, skip_existing: bool):
    """Run full reduction pipeline from config file.

    CONFIG_FILE should be a YAML file with instrument, files, and output settings.

    Example config.yaml:

    \b
        instrument: UVES
        output: /data/reduced/
        mode: RED
        files:
          bias: /data/raw/bias/*.fits
          flat: /data/raw/flat/*.fits
          wavecal: /data/raw/thar/*.fits
          science: /data/raw/science/*.fits
        steps: [bias, flat, trace, wavecal, extract]
    """
    import yaml

    with open(config_file) as f:
        config = yaml.safe_load(f)

    instrument_name = config.get("instrument")
    if not instrument_name:
        raise click.ClickException("Config file must specify 'instrument'")

    inst = load_instrument(instrument_name)
    output = config.get("output", ".")
    mode = config.get("mode", "")
    target = config.get("target", "")
    files = config.get("files", {})
    config_steps = config.get("steps", [])

    # Parse steps
    if steps != "all":
        config_steps = [s.strip() for s in steps.split(",")]
    elif not config_steps:
        config_steps = ["bias", "flat", "trace", "wavecal", "extract"]

    click.echo(f"Running pipeline for {instrument_name}")
    click.echo(f"Steps: {', '.join(config_steps)}")
    click.echo(f"Output: {output}")

    pipe = Pipeline(inst, output, mode=mode, target=target)

    # Add steps based on config
    if "bias" in config_steps and files.get("bias"):
        pipe = pipe.bias(_expand_globs(files["bias"]))

    if "flat" in config_steps and files.get("flat"):
        pipe = pipe.flat(_expand_globs(files["flat"]))

    if "trace" in config_steps:
        trace_files = files.get("orders") or files.get("flat")
        pipe = pipe.trace_orders(_expand_globs(trace_files) if trace_files else None)

    if "scatter" in config_steps:
        pipe = pipe.scatter()

    if "norm_flat" in config_steps:
        pipe = pipe.normalize_flat()

    if "wavecal" in config_steps and files.get("wavecal"):
        pipe = pipe.wavelength_calibration(_expand_globs(files["wavecal"]))

    if "curvature" in config_steps:
        curv_files = files.get("curvature") or files.get("wavecal")
        pipe = pipe.curvature(_expand_globs(curv_files) if curv_files else None)

    if "extract" in config_steps and files.get("science"):
        pipe = pipe.extract(_expand_globs(files["science"]))

    if "continuum" in config_steps:
        pipe = pipe.continuum()

    if "finalize" in config_steps:
        pipe = pipe.finalize()

    pipe.run(skip_existing=skip_existing)
    click.echo("Pipeline complete!")


def _expand_globs(patterns) -> list[str]:
    """Expand glob patterns to file list."""
    if isinstance(patterns, str):
        patterns = [patterns]

    files = []
    for pattern in patterns:
        expanded = glob(pattern)
        if expanded:
            files.extend(expanded)
        else:
            # If no glob match, treat as literal path
            if Path(pattern).exists():
                files.append(pattern)
    return sorted(set(files))


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
