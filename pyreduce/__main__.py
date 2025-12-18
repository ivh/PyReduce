"""
PyReduce command-line interface.

Usage:
    uv run reduce --help
    uv run reduce run UVES HD132205 --night 2010-04-01
    uv run reduce run UVES HD132205 --steps bias,flat,orders
    uv run reduce combine --output combined.fits *.final.fits
"""

import click

from . import datasets
from .configuration import get_configuration_for_instrument
from .reduce import main as reduce_main
from .tools.combine import combine as tools_combine

ALL_STEPS = (
    "bias",
    "flat",
    "orders",
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
@click.argument("target")
@click.option("--night", "-n", default=None, help="Observation night (YYYY-MM-DD)")
@click.option("--arm", "-a", default=None, help="Instrument arm/detector")
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
    "--input-dir", "-i", default="raw", help="Input directory relative to base"
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
def run(
    instrument,
    target,
    night,
    arm,
    steps,
    base_dir,
    input_dir,
    output_dir,
    plot,
    order_range,
):
    """Run the reduction pipeline.

    INSTRUMENT: Name of the instrument (e.g., UVES, HARPS, XSHOOTER)
    TARGET: Target star name or regex pattern
    """
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

    # Run reduction
    reduce_main(
        instrument=instrument,
        target=target,
        night=night,
        arms=arm,
        steps=steps,
        base_dir=base_dir or "",
        input_dir=input_dir,
        output_dir=output_dir,
        configuration=config,
        order_range=order_range,
        plot=plot,
    )


@cli.command()
@click.argument("files", nargs=-1, required=True)
@click.option("--output", "-o", default="combined.fits", help="Output filename")
@click.option("--plot", "-p", default=None, type=int, help="Plot specific order")
def combine(files, output, plot):
    """Combine multiple reduced spectra.

    FILES: Input .final.fits files to combine
    """
    tools_combine(list(files), output, plot=plot)


@cli.command()
@click.argument("instrument")
def download(instrument):
    """Download sample dataset for an instrument.

    INSTRUMENT: Name of the instrument (e.g., UVES, HARPS)
    """
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


@cli.command("list-steps")
def list_steps():
    """List all available reduction steps."""
    click.echo("Available reduction steps:")
    for step in ALL_STEPS:
        click.echo(f"  - {step}")


@cli.group()
def step():
    """Run individual reduction steps."""
    pass


def make_step_command(step_name):
    """Factory to create a command for a single step."""

    @click.command(name=step_name)
    @click.argument("instrument")
    @click.argument("target")
    @click.option("--night", "-n", default=None, help="Observation night")
    @click.option("--arm", "-a", default=None, help="Instrument arm")
    @click.option("--base-dir", "-b", default=None, help="Base directory")
    @click.option("--input-dir", "-i", default="raw", help="Input directory")
    @click.option("--output-dir", "-o", default="reduced", help="Output directory")
    @click.option("--plot", "-p", default=0, help="Plot level")
    def cmd(instrument, target, night, arm, base_dir, input_dir, output_dir, plot):
        config = get_configuration_for_instrument(instrument)
        reduce_main(
            instrument=instrument,
            target=target,
            night=night,
            arms=arm,
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
    step.add_command(make_step_command(_step))


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
