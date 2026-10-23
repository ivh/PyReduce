"""Re-vendor slitdec.{c,h} from a local CharSlit checkout.

    uv run python tools/sync_slitdec.py ../CharSlit.git

Copies both files into pyreduce/clib/, stamps the header with CharSlit's
commit, strips trailing whitespace (our pre-commit hook would do it anyway),
and prints the slitdec() signature diff. It does not touch cwrappers.py or
the numpy/numba ports: those need porting by hand, see CLAUDE.md.
"""

import difflib
import re
import subprocess
import sys
from pathlib import Path

CLIB = Path(__file__).resolve().parent.parent / "pyreduce" / "clib"
NOTE = """/* Copied from CharSlit (github.com/ivh/CharSlit), commit {commit}.
   Plain-C slit-decomposition algorithm, compiled here via CFFI.
   Do not edit here: change CharSlit, then run tools/sync_slitdec.py.
   Only divergence: trailing whitespace is stripped.
   PYREDUCE_EXTRACTION=charslit runs the external charslit package instead. */

"""
NOTE_RE = re.compile(r"\A/\* Copied from CharSlit.*?\*/\n\n", re.S)


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def signature(header):
    m = re.search(r"int slitdec\((.*?)\);", header, re.S)
    return [" ".join(a.split()) for a in m.group(1).split(",")] if m else []


def main(repo):
    repo = Path(repo)
    if git(
        repo, "status", "--porcelain", "--", "slitdec/slitdec.c", "slitdec/slitdec.h"
    ):
        sys.exit(f"{repo}/slitdec has uncommitted changes; commit them first")
    commit = git(repo, "rev-parse", "--short", "HEAD")

    old_sig = signature((CLIB / "slitdec.h").read_text())
    for name in ("slitdec.c", "slitdec.h"):
        text = (repo / "slitdec" / name).read_text()
        text = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
        if name == "slitdec.h":
            text = NOTE.format(commit=commit) + NOTE_RE.sub("", text)
        (CLIB / name).write_text(text)
    new_sig = signature((CLIB / "slitdec.h").read_text())

    print(f"Vendored slitdec.{{c,h}} from CharSlit {commit}")
    if old_sig != new_sig:
        print("\nslitdec() signature changed -- update the call in cwrappers.py:")
        print("\n".join(difflib.unified_diff(old_sig, new_sig, lineterm="", n=1)))
    print(
        "\nNext: uv run reduce-build, port the change to numpy_slitdec.py and "
        "numba_slitdec.py,\nthen run test_numpy_slitdec.py and test_numba_slitdec.py."
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
