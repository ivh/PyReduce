#!/usr/bin/python3

import sys
print(sys.path)

import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as mpl

f = open('dump.bin')
osample,ncols,nrows,ny = 10, 768, 15, 161


dt='double'
sL = np.fromfile(f, count=ny, dtype=dt)
sP = np.fromfile(f, count=ncols, dtype=dt)
sP_old = np.fromfile(f, count=ncols, dtype=dt)
im = np.fromfile(f, count=ncols*nrows, dtype=dt)
model = np.fromfile(f, count=ncols*nrows, dtype=dt)

f.close()

F = mpl.Figure()
ax1=F.add_subplot(1,3,1)

ax1.plot(sL)

mpl.show()
