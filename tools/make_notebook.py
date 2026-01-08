#!/usr/bin/env python
# /// script
# dependencies = ["jupytext", "ipykernel", "nbconvert", "click"]
# ///
"""Convert example scripts to Jupyter notebooks.

Usage:
    uv run scripts/make_notebook.py uves
    uv run scripts/make_notebook.py harps --no-execute
"""

import os
import subprocess
import sys

import click


@click.command()
@click.argument("name")
@click.option("--no-execute", is_flag=True, help="Convert without executing")
def main(name, no_execute):
    """Convert an example script to Jupyter notebook."""
    if name.endswith(".py"):
        name = name[:-3]
    if name.endswith("_example"):
        name = name[:-8]

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    py_file = os.path.join(base, "examples", f"{name}_example.py")
    ipynb_file = os.path.join(base, "examples", f"{name}_example.ipynb")

    if not os.path.exists(py_file):
        raise click.ClickException(f"Example not found: {py_file}")

    cmd = ["jupytext", "--to", "notebook"]
    if not no_execute:
        cmd.append("--execute")
    cmd.extend([py_file, "--output", ipynb_file])

    click.echo(f"Converting {py_file} -> {ipynb_file}")
    result = subprocess.run(cmd, check=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
