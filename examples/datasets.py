import os
import requests
import tarfile

def UVES_HD132205(local_dir=None):
    # load data if necessary
    if local_dir is None:
        local_dir = os.path.dirname(__file__)
    target_dir = os.path.join(local_dir, "datasets")
    filename = os.path.join(local_dir, "uves_data.tar.gz")

    if not os.path.isfile(filename):
        remote_location = (
            "http://www.astro.uu.se/~piskunov/RESEARCH/REDUCE/FTP/reduce_demo.tar.gz"
        )
        r = requests.get(remote_location)
        with open(filename, "wb") as fd:
            fd.write(r.content)


    with tarfile.open(filename) as file:
        file.extractall(path=target_dir)
    
    return local_dir


if __name__ == "__main__":
    UVES_HD132205()
