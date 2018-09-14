from . import util, combine_frames

settings = util.read_config()
git_remote = settings["git.remote"] if "git.remote" in settings.keys() else "origin"
util.checkGitRepo(remote_name=git_remote)
