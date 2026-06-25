# Generating MOSAIC `bundle_centers_<channel>.yaml` semi-manually

`bundle_centers` give the cross-dispersion (y) position of each fiber bundle at
the detector center column. They are a y-prior so the trace step can assign
detected traces to the right bundle (`fibers.bundles.merge: center_weight`).

## Why not automatic detection

Automatic peak-detect + group-by-gaps was tried and abandoned. It is not robust
because:

- Dead/missing fibers open a gap *inside* a bundle that looks like the gap
  *between* bundles, so grouping mis-splits and mis-centers bundles.
- Borrowing another channel's file as a prior does not transfer: the J and H
  cameras (and VIS detectors) have slightly different bundle spacing
  (~1 px/bundle), which compounds to ~100 px over 90 bundles and misassigns the
  outer bundles.

## What works: a few anchors + a low-degree polynomial

The bundle centers vs bundle index are very smooth (a slight curvature, ~20 px
away from linear at mid-detector). A degree-2 polynomial through a handful of
hand-read anchor points reproduces all 90 to sub-pixel accuracy.

### Steps

1. Plot the flat (or its center-column profile) and read off the y of a few
   bundle centers together with their **bundle index**. Bundles are numbered
   **top-down: bundle 1 = highest y**. Enough points: the top 3, two from the
   middle (note which bundle numbers), and the bottom 2.
2. Fit `y = poly(index)` at degree 2.
3. Evaluate for index 1..N (N = number of bundles, 90 for NIR) and write the
   yaml.
4. (Optional) Validate: overlay the fit on the flat profile, or compare to
   detected bundle clusters; expect sub-pixel median error.

### Inferring an unknown middle index

If you do not know a middle anchor's exact bundle number, fit the securely
indexed points first (top + bottom), then take the index whose predicted y is
closest to the middle value. For J_LR the two middle values 2409/2367 came out
exactly at bundles 36/37.

### Reference: J_LR (90 bundles, c01 flat)

Anchors used (index: y):

    1: 3902   2: 3860   3: 3816
    36: 2409  37: 2367
    89: 201   90: 160

```python
import numpy as np, yaml

idx = np.array([1, 2, 3, 36, 37, 89, 90])
y   = np.array([3902, 3860, 3816, 2409, 2367, 201, 160])
c = np.polyfit(idx, y, 2)               # deg-2; RMS ~0.3 px on anchors

n = np.arange(1, 91)
centers = {int(i): round(float(v), 1) for i, v in zip(n, np.polyval(c, n))}

with open("pyreduce/instruments/MOSAIC/bundle_centers_j_lr.yaml", "w") as f:
    f.write("# Bundle centers for MOSAIC J_LR (90 bundles)\n")
    f.write("# Bundle ID -> y at detector center; deg-2 polyfit to manual anchors\n")
    for i in n:
        f.write(f"{int(i)}: {centers[int(i)]}\n")
```

Result for J_LR: median error 0.7 px vs the detected bundle clusters (all 53
clean clusters matched), spacing 41-43 px, b1=3902 down to b90=160.

## Per channel

Repeat per channel with its own flat and anchors. Each NIR mode (J_LR, H_LR,
H_HR) and each VIS `<mode>_<quadrant>` channel needs its own file because the
detectors/cameras differ. The header comment in each file should record the
anchors used so the fit can be reproduced.
