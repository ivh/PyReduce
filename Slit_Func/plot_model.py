#!/usr/bin/python3

import numpy as np
import matplotlib
matplotlib.use('QT4Agg')

import matplotlib.pyplot as plt

f = open('dump.bin')
osample,ncols,nrows,ny = 10, 768, 15, 161


dt='double'
sL = np.fromfile(f, count=ny, dtype=dt)
sP = np.fromfile(f, count=ncols, dtype=dt)
sP_old = np.fromfile(f, count=ncols, dtype=dt)
im = np.fromfile(f, count=ncols*nrows, dtype=dt)
model = np.fromfile(f, count=ncols*nrows, dtype=dt)

foo = np.fromfile(f, count=1, dtype=dt)
print(sL)

f.close()

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3,nrows=1,squeeze=True,fig_kw=dict(figsize=(5,10)))

ax1.plot(sL)


plt.show()
