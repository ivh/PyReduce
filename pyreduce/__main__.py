# -*- coding: utf-8 -*-
"""
This is used when pyreduce is used as a script
"""

import argparse
import sys

from .reduce import main
from .tools.combine import combine as tools_combine

scripts = ["reduce", "combine"]


def help():
    parser = argparse.ArgumentParser(
        description="PyReduce script tools interface", prog="python -m pyreduce"
    )
    parser.add_argument("script", help="which script to execute", choices=scripts)
    parser.print_help()


def reduce():
    parser = argparse.ArgumentParser(
        description="PyReduce script tools interface", prog="python -m pyreduce"
    )
    parser.add_argument("script", help="which script to execute", choices=["reduce"])

    parser.add_argument("-b", "--bias", action="store_true", help="Create master bias")
    parser.add_argument("-f", "--flat", action="store_true", help="Create master flat")
    parser.add_argument("-o", "--orders", action="store_true", help="Trace orders")
    parser.add_argument("-n", "--norm_flat", action="store_true", help="Normalize flat")
    parser.add_argument(
        "-w", "--wavecal", action="store_true", help="Prepare wavelength calibration"
    )
    parser.add_argument(
        "-s", "--science", action="store_true", help="Extract science spectrum"
    )
    parser.add_argument(
        "-c", "--continuum", action="store_true", help="Normalize continuum"
    )

    parser.add_argument("instrument", type=str, help="instrument used")
    parser.add_argument("target", type=str, help="target star")

    args = parser.parse_args()
    instrument = args.instrument.upper()
    target = args.target.upper()

    steps_to_take = {
        "bias": args.bias,
        "flat": args.flat,
        "orders": args.orders,
        "norm_flat": args.norm_flat,
        "wavecal": args.wavecal,
        "science": args.science,
        "continuum": args.continuum,
    }
    steps_to_take = [k for k, v in steps_to_take.items() if v]

    # if no steps are specified use all
    if len(steps_to_take) == 0:
        steps_to_take = "all"

    main(instrument=instrument, target=target, steps=steps_to_take)


def combine():
    parser = argparse.ArgumentParser(
        description="PyReduce script tools interface", prog="python -m pyreduce"
    )
    parser.add_argument("script", help="which script to execute", choices=["combine"])

    parser.add_argument(
        "--output", help="destination of the combined file", default="./combined.ech"
    )
    parser.add_argument(
        "--plot", type=int, help="plot the results of the desired order"
    )

    parser.add_argument(
        "input", nargs="+", help="input files to use", default="./*.final.ech"
    )

    args = parser.parse_args()

    files = args.input
    output = args.output
    plot = args.plot

    tools_combine(files, output, plot=plot)


if __name__ == "__main__":
    # Determine which script to run
    if len(sys.argv) == 1:
        script = "help"
    else:
        script = sys.argv[1]

    # Run the chosen script
    if script == "reduce":
        reduce()
    elif script == "combine":
        combine()
    else:
        help()
