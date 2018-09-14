from .util import read_config
from .util import checkGitRepo

settings = read_config()
git_remote = settings["git.remote"] if "git.remote" in settings.keys() else "origin"
checkGitRepo(remote_name=git_remote)
