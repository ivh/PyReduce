#!/usr/bin/python3

import numpy as np
import matplotlib
matplotlib.use('QT4Agg')

import matplotlib.pyplot as plt
plt.ioff()

def midstep(ax, y):
    return ax.step(np.arange(len(y)), y, where='mid')

f = open('dump.bin')
osample,ncols,nrows,ny = 10, 768, 15, 161


dt='double'
sL = np.fromfile(f, count=ny, dtype=dt)
sP = np.fromfile(f, count=ncols, dtype=dt)
sP_old = np.fromfile(f, count=ncols, dtype=dt)
im = np.fromfile(f, count=ncols*nrows, dtype=dt)
model = np.fromfile(f, count=ncols*nrows, dtype=dt)
model.shape = nrows, ncols

foo = np.fromfile(f, count=1, dtype=dt)

f.close()

fig, (ax1, ax2, ax3) = plt.subplots(ncols=1,nrows=3,squeeze=True,figsize=(13,4))

midstep(ax1,sL)
ax1.set_title('sL')

midstep(ax2,sP)
ax2.set_title('sP')

ax3.imshow(model)
ax3.set_title('model')


plt.show()
