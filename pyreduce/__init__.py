import os
from . import util

util.start_logging(None)

dirname = os.path.dirname(__file__)
fname = os.path.join(dirname, "clib", "_slitfunc_2d.c")

if not os.path.exists(fname):
    from .clib import build_extract

    build_extract.build()

from . import reduce, datasets


# from . import util, reduce

# settings = util.read_config()
# git_remote = settings["git.remote"] if "git.remote" in settings.keys() else "origin"
# util.checkGitRepo(remote_name=git_remote)
