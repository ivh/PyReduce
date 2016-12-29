#!/usr/bin/python3
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.ioff()

def midstep(ax, y):
    return ax.step(np.arange(len(y)), y, where='mid')

fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=1,nrows=4,squeeze=True,figsize=(15,6))
ax1.set_title('sL')
ax2.set_title('sP')
ax3.set_title('model')
ax3.set_title('im-model')

for fname in ['dump.bin', 'dump_lapack.bin']:
    f = open(fname)
    osample,ncols,nrows,ny = 10, 768, 15, 161


    dt=np.float64
    sL = np.fromfile(f, count=ny, dtype=dt)
    sP = np.fromfile(f, count=ncols, dtype=dt)
    sP_old = np.fromfile(f, count=ncols, dtype=dt)
    im = np.fromfile(f, count=ncols*nrows, dtype=dt)
    model = np.fromfile(f, count=ncols*nrows, dtype=dt)
    model.shape = nrows, ncols
    im.shape = model.shape

    foo = np.fromfile(f, count=1, dtype=dt)

    f.close()

    midstep(ax1,sL)
    midstep(ax2,sP)

    ax3.imshow(model)

    ax4.imshow(im-model)

xy=list(ax1.axis())
xy[2] -= (xy[3] - xy[2])/8.
ax1.axis(xy)

#plt.show()
PNG='/tmp/slitfu.png'
plt.savefig(PNG, dpi=80)
os.system('nettemp %s'%PNG)
