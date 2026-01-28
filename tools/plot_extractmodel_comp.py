from glob import glob

import numpy as np
from matplotlib import pyplot as plt

fnames = glob("*.npz")
for fname in fnames:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True)
    m1 = np.load(fname)["model"]
    m2 = np.load("/Users/tom/REDUCE_DATA/debug/" + fname)["model"]
    d = m2 - m1
    ax1.imshow(m1, vmin=np.percentile(m1, 5), vmax=np.percentile(m1, 95), cmap="magma")
    ax2.imshow(d, vmin=-100, vmax=100, cmap="bwr")
    ax3.imshow(np.abs(d) / m1, vmin=0, vmax=0.01, cmap="plasma")
plt.show()
