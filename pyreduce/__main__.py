"""
PyReduce command-line interface.

Usage:
    uv run reduce --help
    uv run reduce run UVES HD132205 --night 2010-04-01
    uv run reduce run UVES HD132205 --steps bias,flat,trace
    uv run reduce trace UVES HD132205
    uv run reduce combine --output combined.fits *.final.fits
"""

import click

ALL_STEPS = (
    "bias",
    "flat",
    "trace",
    "curvature",
    "scatter",
    "norm_flat",
    "wavecal_master",
    "wavecal_init",
    "wavecal",
    "freq_comb_master",
    "freq_comb",
    "science",
    "continuum",
    "finalize",
)


@click.group()
@click.version_option(package_name="pyreduce-astro")
def cli():
    """PyReduce - Echelle spectrograph data reduction pipeline."""
    pass


@cli.command()
@click.argument("instrument")
@click.option("--target", "-t", default=None, help="Target star name or regex pattern")
@click.option("--night", "-n", default=None, help="Observation night (YYYY-MM-DD)")
@click.option("--channel", "-c", default=None, help="Instrument channel")
@click.option(
    "--steps",
    "-s",
    default=None,
    help="Comma-separated steps to run (default: all)",
)
@click.option(
    "--base-dir",
    "-b",
    default=None,
    help="Base directory for data (default: $REDUCE_DATA or ~/REDUCE_DATA)",
)
@click.option(
    "--input-dir", "-i", default=None, help="Input directory relative to base"
)
@click.option(
    "--output-dir", "-o", default="reduced", help="Output directory relative to base"
)
@click.option(
    "--plot", "-p", default=0, help="Plot level (0=none, 1=save, 2=interactive)"
)
@click.option(
    "--order-range",
    default=None,
    help="Order range to process (e.g., '1,21')",
)
@click.option(
    "--settings",
    default=None,
    type=click.Path(exists=True),
    help="JSON file with settings overrides",
)
def run(
    instrument,
    target,
    night,
    channel,
    steps,
    base_dir,
    input_dir,
    output_dir,
    plot,
    order_range,
    settings,
):
    """Run the reduction pipeline.

    INSTRUMENT: Name of the instrument (e.g., UVES, HARPS, XSHOOTER)
    """
    from .configuration import get_configuration_for_instrument, load_settings_override
    from .reduce import main as reduce_main

    # Parse steps
    if steps:
        steps = tuple(s.strip() for s in steps.split(","))
    else:
        steps = "all"

    # Parse order range
    if order_range:
        parts = order_range.split(",")
        order_range = (int(parts[0]), int(parts[1]))

    # Load configuration
    config = get_configuration_for_instrument(instrument)
    if settings:
        config = load_settings_override(config, settings)

    # Run reduction
    try:
        reduce_main(
            instrument=instrument,
            target=target,
            night=night,
            channels=channel,
            steps=steps,
            base_dir=base_dir,
            input_dir=input_dir,
            output_dir=output_dir,
            configuration=config,
            order_range=order_range,
            plot=plot,
        )
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from None


@cli.command()
@click.argument("files", nargs=-1, required=True)
@click.option("--output", "-o", default="combined.fits", help="Output filename")
@click.option("--plot", "-p", default=None, type=int, help="Plot specific order")
def combine(files, output, plot):
    """Combine multiple reduced spectra.

    FILES: Input .final.fits files to combine
    """
    from .tools.combine import combine as tools_combine

    tools_combine(list(files), output, plot=plot)


@cli.command()
@click.argument("instrument")
def download(instrument):
    """Download sample dataset for an instrument.

    INSTRUMENT: Name of the instrument (e.g., UVES, HARPS)
    """
    from . import datasets

    instrument = instrument.upper()
    dataset_func = getattr(datasets, instrument, None)
    if dataset_func is None:
        available = [
            name
            for name in dir(datasets)
            if name.isupper() and not name.startswith("_")
        ]
        raise click.ClickException(
            f"Unknown instrument '{instrument}'. Available: {', '.join(available)}"
        )
    path = dataset_func()
    click.echo(f"Dataset downloaded to: {path}")


@cli.command()
@click.argument("filename", required=False)
@click.option(
    "--list", "-l", "list_examples", is_flag=True, help="List available examples"
)
@click.option("--all", "-a", "download_all", is_flag=True, help="Download all examples")
@click.option("--run", "-r", is_flag=True, help="Run the example after downloading")
@click.option("--output", "-o", default=".", help="Output directory")
def examples(filename, list_examples, download_all, run, output):
    """List, download, or run example scripts from GitHub.

    Downloads examples matching your installed PyReduce version.

    \b
    Examples:
        reduce examples                     # List available examples
        reduce examples uves_example.py     # Download to current dir
        reduce examples -r uves_example.py  # Download and run
        reduce examples --all -o ~/scripts  # Download all to ~/scripts
    """
    import json
    import os
    import subprocess
    import sys
    import tempfile
    import urllib.request
    from urllib.error import HTTPError

    from pyreduce import __version__

    version = __version__.split("+")[0]
    if version == "unknown":
        raise click.ClickException(
            "Cannot determine package version. Install from PyPI or a tagged release."
        )

    github_api = (
        f"https://api.github.com/repos/ivh/PyReduce/contents/examples?ref=v{version}"
    )
    github_raw = f"https://raw.githubusercontent.com/ivh/PyReduce/v{version}/examples"

    # Fetch list of examples from GitHub API
    try:
        with urllib.request.urlopen(github_api) as resp:
            contents = json.loads(resp.read().decode())
    except HTTPError as e:
        if e.code == 404:
            raise click.ClickException(
                f"Tag v{version} not found on GitHub. "
                "Try installing a released version."
            ) from None
        raise click.ClickException(f"GitHub API error: {e}") from None

    example_files = sorted(f["name"] for f in contents if f["name"].endswith(".py"))

    # List mode
    if list_examples or (not filename and not download_all):
        click.echo(f"Available examples for v{version}:")
        for name in example_files:
            click.echo(f"  {name}")
        return

    if run and download_all:
        raise click.ClickException("Cannot use --run with --all")

    # Run mode: download to temp and execute
    if run:
        if filename not in example_files:
            raise click.ClickException(
                f"Unknown example '{filename}'. Use 'reduce examples --list' to see available."
            )
        url = f"{github_raw}/{filename}"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            try:
                with urllib.request.urlopen(url) as resp:
                    f.write(resp.read().decode())
                temp_path = f.name
            except HTTPError as e:
                raise click.ClickException(
                    f"Failed to download {filename}: {e}"
                ) from None
        try:
            click.echo(f"Running {filename}...")
            result = subprocess.run([sys.executable, temp_path], check=False)
            sys.exit(result.returncode)
        finally:
            os.unlink(temp_path)

    # Ensure output directory exists
    os.makedirs(output, exist_ok=True)

    def download_file(name):
        url = f"{github_raw}/{name}"
        dest = os.path.join(output, name)
        try:
            urllib.request.urlretrieve(url, dest)
            click.echo(f"Downloaded: {dest}")
        except HTTPError as e:
            click.echo(f"Failed to download {name}: {e}", err=True)

    if download_all:
        for name in example_files:
            download_file(name)
    else:
        if filename not in example_files:
            raise click.ClickException(
                f"Unknown example '{filename}'. Use 'reduce examples --list' to see available."
            )
        download_file(filename)


@cli.command("list-steps")
def list_steps():
    """List all available reduction steps."""
    click.echo("Available reduction steps:")
    for step in ALL_STEPS:
        click.echo(f"  - {step}")


def make_step_command(step_name):
    """Factory to create a command for a single step."""

    @click.command(name=step_name)
    @click.argument("instrument")
    @click.argument("target", required=False, default="")
    @click.option("--night", "-n", default=None, help="Observation night")
    @click.option("--channel", "-c", default=None, help="Instrument channel")
    @click.option("--base-dir", "-b", default=None, help="Base directory")
    @click.option("--input-dir", "-i", default="raw", help="Input directory")
    @click.option("--output-dir", "-o", default="reduced", help="Output directory")
    @click.option("--plot", "-p", default=0, help="Plot level")
    @click.option(
        "--file",
        "-f",
        default=None,
        help="Specific input file (bypasses file discovery)",
    )
    @click.option(
        "--settings",
        default=None,
        type=click.Path(exists=True),
        help="JSON file with settings overrides",
    )
    def cmd(
        instrument,
        target,
        night,
        channel,
        base_dir,
        input_dir,
        output_dir,
        plot,
        file,
        settings,
    ):
        from .configuration import (
            get_configuration_for_instrument,
            load_settings_override,
        )
        from .reduce import main as reduce_main

        if file:
            # Direct file mode: run step on specific file
            import os

            import numpy as np

            from . import reduce as reduce_module
            from .instruments.instrument_info import load_instrument

            inst = load_instrument(instrument)
            channel = channel or (inst.channels[0] if inst.channels else "")
            output_dir_full = output_dir
            if base_dir:
                output_dir_full = os.path.join(base_dir, output_dir)
            os.makedirs(output_dir_full, exist_ok=True)

            # Load configuration for this step
            config = get_configuration_for_instrument(instrument)
            if settings:
                config = load_settings_override(config, settings)

            # Step classes that support --file (take raw files as input)
            step_classes = {
                "bias": reduce_module.Bias,
                "flat": reduce_module.Flat,
                "trace": reduce_module.OrderTracing,
                "curvature": reduce_module.SlitCurvatureDetermination,
                "scatter": reduce_module.BackgroundScatter,
                "wavecal_master": reduce_module.WavelengthCalibrationMaster,
                "freq_comb_master": reduce_module.LaserFrequencyCombMaster,
                "science": reduce_module.ScienceExtraction,
            }

            # Steps that don't take raw files - --file makes no sense
            no_file_steps = {
                "mask",
                "norm_flat",
                "wavecal_init",
                "wavecal",
                "freq_comb",
                "continuum",
                "finalize",
                "rectify",
            }

            if step_name in no_file_steps:
                raise click.ClickException(
                    f"Step '{step_name}' does not accept raw files. "
                    f"Run without --file to use the pipeline."
                )

            if step_name not in step_classes:
                raise click.ClickException(
                    f"Step '{step_name}' does not support --file option"
                )

            def make_step(name):
                """Create a step instance."""
                step_class = {
                    "mask": reduce_module.Mask,
                    "bias": reduce_module.Bias,
                    "flat": reduce_module.Flat,
                    "trace": reduce_module.OrderTracing,
                    "curvature": reduce_module.SlitCurvatureDetermination,
                    "scatter": reduce_module.BackgroundScatter,
                    "norm_flat": reduce_module.NormalizeFlatField,
                    "wavecal_master": reduce_module.WavelengthCalibrationMaster,
                    "freq_comb_master": reduce_module.LaserFrequencyCombMaster,
                    "science": reduce_module.ScienceExtraction,
                }.get(name)
                if not step_class:
                    return None
                step_config = config.get(name, {}).copy()
                step_config["plot"] = 0  # No plots for dependency loading
                return step_class(
                    inst,
                    channel,
                    target=target or "",
                    night=night,
                    output_dir=output_dir_full,
                    order_range=None,
                    **step_config,
                )

            def load_dependency(name, loaded):
                """Load a dependency from disk."""
                if name in loaded:
                    return loaded[name]
                if name == "config":
                    loaded["config"] = config
                    return config
                dep_step = make_step(name)
                if dep_step is None:
                    return None
                # Load this step's dependencies first
                for sub_dep in dep_step.loadDependsOn:
                    load_dependency(sub_dep, loaded)
                # Build kwargs for load()
                load_kwargs = {
                    d: loaded[d] for d in dep_step.loadDependsOn if d in loaded
                }
                try:
                    loaded[name] = dep_step.load(**load_kwargs)
                except FileNotFoundError:
                    loaded[name] = None
                return loaded[name]

            # Create the target step
            step_config = config.get(step_name, {}).copy()
            step_config["plot"] = plot
            step_class = step_classes[step_name]
            step = step_class(
                inst,
                channel,
                target=target or "",
                night=night,
                output_dir=output_dir_full,
                order_range=None,
                **step_config,
            )

            # Load all dependencies
            loaded = {}
            for dep in step.dependsOn:
                load_dependency(dep, loaded)

            # Build kwargs for run()
            run_kwargs = {d: loaded.get(d) for d in step.dependsOn}
            run_kwargs["files"] = np.array([file])

            step.run(**run_kwargs)
        else:
            config = get_configuration_for_instrument(instrument)
            if settings:
                config = load_settings_override(config, settings)
            reduce_main(
                instrument=instrument,
                target=target,
                night=night,
                channels=channel,
                steps=(step_name,),
                base_dir=base_dir or "",
                input_dir=input_dir,
                output_dir=output_dir,
                configuration=config,
                plot=plot,
            )

    cmd.__doc__ = f"Run the '{step_name}' step."
    return cmd


# Register individual step commands
for _step in ALL_STEPS:
    cli.add_command(make_step_command(_step))


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
