"""Module for extracting data from observations

Authors
-------

Version
-------

License
-------
"""

import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
from scipy.interpolate import interp1d
from tqdm import tqdm

from . import util
from .util import make_index

logger = logging.getLogger(__name__)

# Backend selection: set PYREDUCE_USE_CHARSLIT=1 to use charslit
USE_CHARSLIT = os.environ.get("PYREDUCE_USE_CHARSLIT", "0") == "1"

if USE_CHARSLIT:
    import charslit

    logger.info("Using charslit extraction backend")
else:
    from . import cwrappers

    logger.info("Using CFFI extraction backend")


def _slitdec_charslit(
    img,
    ycen,
    slitcurve,
    slitdeltas,
    lambda_sp,
    lambda_sf,
    osample,
    yrange,
    maxiter,
    gain,
    reject_threshold,
    preset_slitfunc,
):
    """Call charslit.slitdec and convert results to the expected format.

    Parameters
    ----------
    img : array[nrows, ncols]
        Input image swath (may be masked array)
    ycen : array[ncols]
        Trace center positions (fractional)
    slitcurve : array[ncols, 6]
        Polynomial coefficients for slit curvature (c0..c5)
    slitdeltas : array[nrows] or None
        Per-row residual offsets
    lambda_sp : float
        Spectrum smoothing parameter
    lambda_sf : float
        Slit function smoothing parameter
    osample : int
        Oversampling factor
    yrange : tuple[int, int]
        Extraction range (below, above)
    maxiter : int
        Maximum iterations
    gain : float
        Detector gain
    reject_threshold : float
        Outlier rejection threshold (not used by charslit currently)
    preset_slitfunc : array or None
        Preset slit function (not supported by charslit yet, ignored)

    Returns
    -------
    sp : array[ncols]
        Extracted spectrum
    sl : array[nslitf]
        Slit function
    model : array[nrows, ncols]
        Model image
    unc : array[ncols]
        Spectrum uncertainties
    mask : array[nrows, ncols]
        Output mask (True = bad)
    info : array[5]
        [success, chi2, status, niter, delta_x]
    """
    nrows, ncols = img.shape

    # Get data and mask
    mask_in = np.ma.getmaskarray(img)
    data = np.ma.getdata(img).astype(np.float64)
    data[~np.isfinite(data)] = 0
    mask_in = mask_in | ~np.isfinite(data)

    # Compute pixel uncertainties (shot noise)
    pix_unc = np.abs(data) * gain
    np.sqrt(pix_unc, out=pix_unc)
    pix_unc[pix_unc < 1] = 1
    pix_unc = pix_unc.astype(np.float64)

    # Convert mask: numpy (True=bad) -> charslit (0=bad, 1=good)
    mask_c = np.where(mask_in, 0, 1).astype(np.uint8)

    # Ensure contiguous arrays
    data = np.ascontiguousarray(data)
    pix_unc = np.ascontiguousarray(pix_unc)
    mask_c = np.ascontiguousarray(mask_c)

    # charslit expects full ycen and does the integer/fractional split internally
    ycen_c = np.ascontiguousarray(ycen.astype(np.float64))

    # charslit expects slitcurve of shape (ncols, 6) - coeffs c0..c5
    slitcurve_c = np.ascontiguousarray(slitcurve.astype(np.float64))

    if slitdeltas is None:
        slitdeltas = np.zeros(nrows, dtype=np.float64)
    slitdeltas = np.ascontiguousarray(slitdeltas.astype(np.float64))

    # Note: preset_slitfunc is not currently supported by charslit
    if preset_slitfunc is not None:
        logger.debug("preset_slitfunc is not yet supported by charslit, ignoring")

    # Call charslit
    result = charslit.slitdec(
        data,
        pix_unc,
        mask_c,
        ycen_c,
        slitcurve_c,
        slitdeltas,
        osample=osample,
        lambda_sP=float(lambda_sp),
        lambda_sL=float(lambda_sf),
        maxiter=maxiter,
    )

    sp = result["spectrum"]
    sl = result["slitfunction"]
    model = result["model"]
    unc = result["uncertainty"]
    return_code = result.get("return_code", 0)
    info_arr = result.get("info", np.zeros(5))

    # Convert mask back: charslit -> numpy (True=bad)
    mask_out = result.get("mask", mask_c)
    mask_out = mask_out == 0

    # Build info array: charslit returns info as [success, cost, status, iter, delta_x]
    if isinstance(info_arr, np.ndarray) and len(info_arr) >= 5:
        info = info_arr
    else:
        info = np.array([float(return_code == 0), 0.0, float(return_code), 0.0, 0.0])

    return sp, sl, model, unc, mask_out, info


def _slitdec_cffi(
    img,
    ycen,
    curvature,
    lambda_sp,
    lambda_sf,
    osample,
    yrange,
    maxiter,
    gain,
    reject_threshold,
    preset_slitfunc,
):
    """Call CFFI slitfunc_curved and return results in the same format as charslit.

    This is the legacy extraction backend using the CFFI C extension.
    Only supports curvature degrees 1-2 (p1, p2).
    """
    # Extract p1, p2 from curvature array
    if curvature is not None:
        p1 = curvature[:, 1] if curvature.shape[1] > 1 else np.zeros(curvature.shape[0])
        p2 = curvature[:, 2] if curvature.shape[1] > 2 else np.zeros(curvature.shape[0])
    else:
        ncols = len(ycen)
        p1 = np.zeros(ncols)
        p2 = np.zeros(ncols)

    sp, sl, model, unc, mask, info = cwrappers.slitfunc_curved(
        img,
        ycen,
        p1,
        p2,
        lambda_sp,
        lambda_sf,
        osample,
        yrange,
        maxiter=maxiter,
        gain=gain,
        reject_threshold=reject_threshold,
        preset_slitfunc=preset_slitfunc,
    )

    return sp, sl, model, unc, mask, info


def _ensure_slitcurve(curvature, ncols, n_coeffs=6):
    """Ensure curvature is in the right format for charslit.

    Parameters
    ----------
    curvature : array[ncols, n_coeffs] or None
        Curvature coefficients for this trace/swath, or None for vertical extraction.
    ncols : int
        Number of columns (for validation/creation if None).
    n_coeffs : int
        Number of coefficients (default 6 for charslit).

    Returns
    -------
    slitcurve : array[ncols, n_coeffs]
        Polynomial coefficients padded to n_coeffs.
    """
    if curvature is None:
        return np.zeros((ncols, n_coeffs), dtype=np.float64)

    curvature = np.asarray(curvature, dtype=np.float64)
    if curvature.shape[1] >= n_coeffs:
        return curvature[:, :n_coeffs]

    # Pad with zeros
    result = np.zeros((ncols, n_coeffs), dtype=np.float64)
    result[:, : curvature.shape[1]] = curvature
    return result


class ProgressPlot:  # pragma: no cover
    def __init__(self, nrow, ncol, nslitf, title=None):
        self.nrow = nrow
        self.ncol = ncol
        self.nslitf = nslitf

        # Setup debug output directory for saving swath data
        from pathlib import Path

        reduce_data = os.environ.get("REDUCE_DATA", os.path.expanduser("~/REDUCE_DATA"))
        self.save_dir = Path(reduce_data) / "debug"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.min_frame_time = float(
            os.environ.get("PYREDUCE_PLOT_ANIMATION_SPEED", 0.3)
        )
        self.last_frame_time = None

        plt.ion()
        plt.rcParams["figure.raise_window"] = False
        self.fig = plt.figure(figsize=(12, 8))

        gs = self.fig.add_gridspec(
            4,
            5,
            height_ratios=[1, 1, 1, 1.2],
            width_ratios=[0.03, 1, 1, 1, 0.8],
            hspace=0.05,
            wspace=0.05,
        )

        # Colorbar axes (left column)
        self.ax_cbar_img = self.fig.add_subplot(gs[0:2, 0])
        self.ax_cbar_resid = self.fig.add_subplot(gs[2, 0])

        # Image panels (stacked vertically with no gaps, no tick labels)
        self.ax_obs = self.fig.add_subplot(gs[0, 1:4])
        self.ax_obs.set_axis_off()

        self.ax_model = self.fig.add_subplot(
            gs[1, 1:4], sharex=self.ax_obs, sharey=self.ax_obs
        )
        self.ax_model.set_axis_off()

        self.ax_resid = self.fig.add_subplot(
            gs[2, 1:4], sharex=self.ax_obs, sharey=self.ax_obs
        )
        self.ax_resid.set_axis_off()

        # Slit function panel (rightmost column, top 3 rows, rotated axes)
        self.ax_slit = self.fig.add_subplot(gs[0:3, 4])
        self.ax_slit.set_title("Slit")
        self.ax_slit.set_xlabel("contribution")
        self.ax_slit.set_ylim((0, nrow))
        self.ax_slit.yaxis.set_label_position("right")
        self.ax_slit.yaxis.tick_right()

        # Spectrum panel (full bottom row)
        self.ax_spec = self.fig.add_subplot(gs[3, 1:])
        self.ax_spec.set_xlim((0, ncol))

        self.title = title
        if title is not None:
            self.fig.suptitle(title)

        # Create image plots
        img = np.ones((nrow, ncol))
        self.im_obs = self.ax_obs.imshow(img, aspect="auto", origin="lower")
        self.im_model = self.ax_model.imshow(img, aspect="auto", origin="lower")
        self.im_resid = self.ax_resid.imshow(
            np.zeros((nrow, ncol)), aspect="auto", origin="lower", cmap="bwr"
        )

        # Colorbars in dedicated axes (ticks/labels on left)
        self.cbar_img = self.fig.colorbar(
            self.im_obs, cax=self.ax_cbar_img, ticklocation="left"
        )
        self.cbar_resid = self.fig.colorbar(
            self.im_resid, cax=self.ax_cbar_resid, ticklocation="left"
        )

        # Spectrum plot elements (rejected first as background, then good points)
        (self.rejected_spec,) = self.ax_spec.plot([], [], ".r", ms=2, alpha=0.2)
        (self.good_spec,) = self.ax_spec.plot([], [], ".g", ms=2, alpha=0.2)
        (self.line_spec,) = self.ax_spec.plot([], "-k")

        # Slit function plot elements (rejected first as background, then good points)
        (self.rejected_slit,) = self.ax_slit.plot([], [], ".r", ms=2, alpha=0.2)
        (self.good_slit,) = self.ax_slit.plot([], [], ".g", ms=2, alpha=0.2)
        (self.line_slit,) = self.ax_slit.plot([], [], "-k", lw=2)

        self.paused = False
        self.advance_one = False
        ax_slower = self.fig.add_axes([0.30, 0.02, 0.08, 0.04])
        ax_faster = self.fig.add_axes([0.39, 0.02, 0.08, 0.04])
        ax_pause = self.fig.add_axes([0.48, 0.02, 0.08, 0.04])
        ax_step = self.fig.add_axes([0.57, 0.02, 0.08, 0.04])
        self.btn_slower = Button(ax_slower, "Slower")
        self.btn_faster = Button(ax_faster, "Faster")
        self.btn_pause = Button(ax_pause, "Pause")
        self.btn_step = Button(ax_step, "Step")
        self.btn_slower.on_clicked(self._slower)
        self.btn_faster.on_clicked(self._faster)
        self.btn_pause.on_clicked(self._toggle_pause)
        self.btn_step.on_clicked(self._step)

        self.fig.subplots_adjust(bottom=0.12, top=0.92, left=0.05, right=0.92)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _slower(self, event=None):
        self.min_frame_time = min(2.0, self.min_frame_time * 1.5)

    def _faster(self, event=None):
        self.min_frame_time = max(0.01, self.min_frame_time / 1.5)

    def _toggle_pause(self, event=None):
        self.paused = not self.paused
        self.btn_pause.label.set_text("Resume" if self.paused else "Pause")
        self.fig.canvas.draw()

    def _step(self, event=None):
        if self.paused:
            self.advance_one = True

    def wait_if_paused(self):
        while self.paused and not self.advance_one:
            self.fig.canvas.flush_events()
            time.sleep(0.05)
        self.advance_one = False

    def plot(
        self,
        img,
        spec,
        slitf,
        model,
        ycen,
        input_mask,
        output_mask,
        trace_idx=0,
        left=0,
        right=0,
        unc=None,
        info=None,
        swath_idx=0,
        save=True,
    ):
        # Save swath data to debug directory
        if save:
            outfile = self.save_dir / f"swath_trace{trace_idx}_swath{swath_idx}.npz"
            np.savez(
                outfile,
                swath_img=img,
                ycen=ycen,
                spec=spec,
                slitf=slitf,
                model=model,
                unc=unc,
                input_mask=input_mask,
                output_mask=output_mask,
                info=info,
            )

        img = np.copy(img)
        spec = np.copy(spec)
        slitf = np.copy(slitf)
        ycen = np.copy(ycen)

        ny = img.shape[0]
        nspec = img.shape[1]
        x_spec, y_spec = self.get_spec(img, spec, slitf, ycen)
        x_slit, y_slit = self.get_slitf(img, spec, slitf, ycen)
        ycen = ycen + ny / 2

        old = np.linspace(-1, ny, len(slitf))

        # Separate rejected (output_mask=True) and good (output_mask=False) pixels
        rejected = output_mask.ravel()
        good = ~output_mask.ravel()

        rej_spec_x = x_spec[rejected]
        rej_spec_y = y_spec[rejected]
        rej_slit_x = x_slit[rejected]
        rej_slit_y = y_slit[rejected]
        good_spec_x = x_spec[good]
        good_spec_y = y_spec[good]
        good_slit_x = x_slit[good]
        good_slit_y = y_slit[good]

        # Update image data
        vmin, vmax = np.percentile(img, [5, 95])
        self.im_obs.set_data(img)
        self.im_obs.set_clim(vmin, vmax)

        # Show masks on model panel: input mask (white), newly rejected (red)
        # Masks are drawn first (underneath), model on top as masked array
        new_bad = output_mask & ~input_mask

        # Create RGBA arrays (transparent where no mask)
        input_rgba = np.zeros((*img.shape, 4), dtype=np.float32)
        input_rgba[input_mask, :] = [1, 1, 1, 1]  # white

        new_rgba = np.zeros((*img.shape, 4), dtype=np.float32)
        new_rgba[new_bad, :] = [1, 0, 0, 1]  # red

        # Match extent of underlying model image
        extent = self.im_model.get_extent()

        if hasattr(self, "_mask_im_new"):
            self._mask_im_new.set_data(new_rgba)
            self._mask_im_input.set_data(input_rgba)
        else:
            # Draw masks first (lower zorder), then model on top
            self._mask_im_new = self.ax_model.imshow(
                new_rgba,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                extent=extent,
                zorder=1,
            )
            self._mask_im_input = self.ax_model.imshow(
                input_rgba,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                extent=extent,
                zorder=2,
            )
            # Move model image to top so cursor reads model values
            self.im_model.set_zorder(3)

        # Model as masked array: transparent where either mask is set
        union_mask = input_mask | output_mask
        model_masked = np.ma.array(model, mask=union_mask)
        self.im_model.set_data(model_masked)
        self.im_model.set_clim(vmin, vmax)

        resid = img - model
        rlim = np.nanpercentile(np.abs(resid), 99)
        self.im_resid.set_data(resid)
        self.im_resid.set_clim(-rlim, rlim)

        # Update spectrum panel
        self.rejected_spec.set_xdata(rej_spec_x)
        self.rejected_spec.set_ydata(rej_spec_y)
        self.good_spec.set_xdata(good_spec_x)
        self.good_spec.set_ydata(good_spec_y)
        self.line_spec.set_xdata(np.arange(len(spec)))
        self.line_spec.set_ydata(spec)

        # Update slit function panel (rotated: contribution on x, y-pixel on y)
        self.rejected_slit.set_xdata(rej_slit_y)
        self.rejected_slit.set_ydata(rej_slit_x)
        self.good_slit.set_xdata(good_slit_y)
        self.good_slit.set_ydata(good_slit_x)
        self.line_slit.set_xdata(slitf)
        self.line_slit.set_ydata(old)

        self.ax_spec.set_xlim((0, nspec - 1))
        spec_middle = spec[5:-5] if len(spec) > 10 else spec
        limit = np.nanmax(spec_middle) * 1.1 if len(spec_middle) > 0 else 1.0
        if not np.isnan(limit):
            self.ax_spec.set_ylim((0, limit))

        self.ax_slit.set_ylim((0, ny - 1))
        limit = np.nanmax(slitf) * 1.1
        if not np.isnan(limit):
            self.ax_slit.set_xlim((0, limit))

        niter = int(info[3]) if info is not None else 0
        title = f"Trace {trace_idx}, Swath {swath_idx}, Columns {left}-{right}, Iter {niter}"
        if self.title is not None:
            title = f"{self.title}\n{title}"
        self.fig.suptitle(title)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        if self.last_frame_time is not None:
            elapsed = time.monotonic() - self.last_frame_time
            remaining = self.min_frame_time - elapsed
            if remaining > 0:
                plt.pause(remaining)
        self.last_frame_time = time.monotonic()

        self.wait_if_paused()

    def close(self):
        plt.ioff()
        plt.close()

    def get_spec(self, img, spec, slitf, ycen):
        """get the spectrum corrected by the slit function"""
        nrow, ncol = img.shape
        x, y = np.indices(img.shape)
        ycen = ycen - ycen.astype(int)

        x = x - ycen + 0.5
        old = np.linspace(-1, nrow - 1 + 1, len(slitf))
        sf = np.interp(x, old, slitf)

        x = img / sf

        x = x.ravel()
        y = y.ravel()
        return y, x

    def get_slitf(self, img, spec, slitf, ycen):
        """get the slit function"""
        x = np.indices(img.shape)[0]
        ycen = ycen - ycen.astype(int)

        if np.any(spec == 0):
            i = np.arange(len(spec))
            try:
                spec = interp1d(
                    i[spec != 0], spec[spec != 0], fill_value="extrapolate"
                )(i)
            except ValueError:
                spec[spec == 0] = np.median(spec)
        y = img / spec[None, :]
        y = y.ravel()

        x = x - ycen + 0.5
        x = x.ravel()
        return x, y


class Swath:
    def __init__(self, nswath):
        self.nswath = nswath
        self.spec = [None] * nswath
        self.slitf = [None] * nswath
        self.model = [None] * nswath
        self.unc = [None] * nswath
        self.mask = [None] * nswath
        self.info = [None] * nswath

    def __len__(self):
        return self.nswath

    def __getitem__(self, key):
        return (
            self.spec[key],
            self.slitf[key],
            self.model[key],
            self.unc[key],
            self.mask[key],
            self.info[key],
        )

    def __setitem__(self, key, value):
        self.spec[key] = value[0]
        self.slitf[key] = value[1]
        self.model[key] = value[2]
        self.unc[key] = value[3]
        self.mask[key] = value[4]
        self.info[key] = value[5]


def fix_parameters(xwd, cr, traces, nrow, ncol, ntrace, ignore_column_range=False):
    """Fix extraction width and column range, so that all pixels used are within the image.
    I.e. the column range is cut so that the everything is within the image

    Parameters
    ----------
    xwd : float
        Total extraction height. Split evenly above/below trace. Values below 3 are fractions of trace spacing.
    cr : 2-tuple(int), array
        Column range, either one value for all traces, or the whole array
    traces : array
        polynomial coefficients that describe each trace
    nrow : int
        Number of rows in the image
    ncol : int
        Number of columns in the image
    ntrace : int
        Number of traces in the image
    ignore_column_range : bool, optional
        if true does not change the column range, however this may lead to problems with the extraction, by default False

    Returns
    -------
    xwd : array
        fixed extraction width
    cr : array
        fixed column range
    traces : array
        the same traces as before
    """

    if xwd is None:
        xwd = 1.0
    if np.isscalar(xwd):
        xwd = np.full(ntrace, xwd)
    else:
        xwd = np.asarray(xwd)
        if xwd.ndim == 1:
            if len(xwd) != ntrace:
                raise ValueError(
                    f"extraction_height array length {len(xwd)} doesn't match ntrace {ntrace}"
                )
        else:
            raise ValueError("extraction_height must be a scalar or 1D array")

    if cr is None:
        cr = np.tile([0, ncol], (ntrace, 1))
    else:
        cr = np.asarray(cr)
        if cr.ndim == 1:
            cr = np.tile(cr, (ntrace, 1))

    traces = np.asarray(traces)

    xwd = np.array([xwd[0], *xwd, xwd[-1]])
    cr = np.array([cr[0], *cr, cr[-1]])
    traces = extend_traces(traces, nrow)

    xwd = fix_extraction_height(xwd, traces, cr, ncol)
    if not ignore_column_range:
        cr, traces = fix_column_range(cr, traces, xwd, nrow, ncol)

    traces = traces[1:-1]
    xwd = xwd[1:-1]
    cr = cr[1:-1]

    return xwd, cr, traces


def extend_traces(traces, nrow):
    """Extrapolate extra traces above and below the existing ones

    Parameters
    ----------
    traces : array[ntrace, degree]
        trace polynomial coefficients
    nrow : int
        number of rows in the image

    Returns
    -------
    traces : array[ntrace + 2, degree]
        extended traces
    """

    ntrace, ncoef = traces.shape

    if ntrace > 1:
        trace_low = 2 * traces[0] - traces[1]
        trace_high = 2 * traces[-1] - traces[-2]
    else:
        trace_low = [0 for _ in range(ncoef)]
        trace_high = [0 for _ in range(ncoef - 1)] + [nrow]

    return np.array([trace_low, *traces, trace_high])


def fix_extraction_height(xwd, traces, cr, ncol):
    """Convert fractional extraction height to pixel range.

    Fractions (< 2) are multiplied by the minimum distance to neighboring traces.

    Parameters
    ----------
    xwd : array[ntrace]
        extraction full height per trace
    traces : array[ntrace, degree]
        trace polynomial coefficients
    cr : array[ntrace, 2]
        column range to use
    ncol : int
        number of columns in image

    Returns
    -------
    xwd : array[ntrace]
        updated extraction full height in pixels
    """

    if not np.all(xwd >= 2):
        x = np.arange(ncol)
        for i in range(1, len(xwd) - 1):
            if xwd[i] < 2:
                # Find minimum distance to neighboring traces
                min_dist = np.inf
                for k in [i - 1, i + 1]:
                    left = max(cr[[i, k], 0])
                    right = min(cr[[i, k], 1])

                    if right < left:
                        raise ValueError(
                            f"Check your column ranges. Traces {i} and {k} are weird"
                        )

                    current = np.polyval(traces[i], x[left:right])
                    neighbor = np.polyval(traces[k], x[left:right])
                    min_dist = min(min_dist, np.min(np.abs(current - neighbor)))

                xwd[i] *= min_dist

        xwd[0] = xwd[1]
        xwd[-1] = xwd[-2]

    xwd = np.ceil(xwd).astype(int)

    return xwd


def fix_column_range(column_range, traces, extraction_height, nrow, ncol):
    """Fix the column range, so that no pixels outside the image will be accessed (Thus avoiding errors)

    Parameters
    ----------
    img : array[nrow, ncol]
        image
    traces : array[ntrace, degree]
        trace polynomial coefficients
    extraction_height : array[ntrace]
        extraction full height in pixels
    column_range : array[ntrace, 2]
        current column range
    no_clip : bool, optional
        if False, new column range will be smaller or equal to current column range, otherwise it can also be larger (default: False)

    Returns
    -------
    column_range : array[ntrace, 2]
        updated column range
    traces : array[ntrace, degree]
        trace polynomial coefficients (may have rows removed if no valid pixels)
    """

    ix = np.arange(ncol)
    to_remove = []
    half = extraction_height / 2
    # Loop over non extension traces
    for i, trace in zip(range(1, len(traces) - 1), traces[1:-1], strict=False):
        # Shift trace up/down by half extraction_height
        coeff_bot, coeff_top = np.copy(trace), np.copy(trace)
        coeff_bot[-1] -= half[i]
        coeff_top[-1] += half[i]

        y_bot = np.polyval(coeff_bot, ix)  # low edge of arc
        y_top = np.polyval(coeff_top, ix)  # high edge of arc

        # find regions of pixels inside the image
        # then use the region that most closely resembles the existing column range (from tracing)
        # but clip it to the existing column range (trace polynomials are not well defined outside the original range)
        points_in_image = np.where((y_bot >= 0) & (y_top < nrow))[0]

        if len(points_in_image) == 0:
            # print(y_bot, y_top,nrow, ncol, points_in_image)
            logger.warning(
                f"No columns are completely within the specified height for trace {i - 1}, removing it."
            )
            to_remove += [i]
            continue

        regions = np.where(np.diff(points_in_image) != 1)[0]
        regions = [(r, r + 1) for r in regions]
        regions = [
            points_in_image[0],
            *points_in_image[(regions,)].ravel(),
            points_in_image[-1],
        ]
        regions = [[regions[i], regions[i + 1] + 1] for i in range(0, len(regions), 2)]
        overlap = [
            min(reg[1], column_range[i, 1]) - max(reg[0], column_range[i, 0])
            for reg in regions
        ]
        iregion = np.argmax(overlap)
        column_range[i] = np.clip(
            regions[iregion], column_range[i, 0], column_range[i, 1]
        )

    column_range[0] = column_range[1]
    column_range[-1] = column_range[-2]

    if to_remove:
        column_range = np.delete(column_range, to_remove, axis=0)
        traces = np.delete(traces, to_remove, axis=0)

    return column_range, traces


def make_bins(swath_width, xlow, xhigh, ycen):
    """Create bins for the swathes
    Bins are roughly equally sized, have roughly length swath width (if given)
    and overlap roughly half-half with each other

    Parameters
    ----------
    swath_width : {int, None}
        initial value for the swath_width, bins will have roughly that size, but exact value may change
        if swath_width is None, determine a good value, from the data
    xlow : int
        lower bound for x values
    xhigh : int
        upper bound for x values
    ycen : array[ncol]
        center of the order trace

    Returns
    -------
    nbin : int
        number of bins
    bins_start : array[nbin]
        left(beginning) side of the bins
    bins_end : array[nbin]
        right(ending) side of the bins
    """

    if swath_width is None:
        ncol = len(ycen)
        i = np.unique(ycen.astype(int))  # Points of row crossing
        # ni = len(i)  # This is how many times this order crosses to the next row
        if len(i) > 1:  # Curved order crosses rows
            i = np.sum(i[1:] - i[:-1]) / (len(i) - 1)
            nbin = np.clip(
                int(np.round(ncol / i)) // 3, 3, 20
            )  # number of swaths along the order
        else:  # Perfectly aligned orders
            nbin = np.clip(ncol // 400, 3, None)  # Still follow the changes in PSF
        nbin = nbin * (xhigh - xlow) // ncol  # Adjust for the true order length
    else:
        nbin = np.clip(int(np.round((xhigh - xlow) / swath_width)), 1, None)

    bins = np.linspace(xlow, xhigh, 2 * nbin + 1)  # boundaries of bins
    bins_start = np.ceil(bins[:-2]).astype(int)  # beginning of each bin
    bins_end = np.floor(bins[2:]).astype(int)  # end of each bin

    return nbin, bins_start, bins_end


def calc_telluric_correction(telluric, img):  # pragma: no cover
    """Calculate telluric correction

    If set to specific integer larger than 1 is used as the
    offset from the order center line. The sky is then estimated by computing
    median signal between this offset and the upper/lower limit of the
    extraction window.

    Parameters
    ----------
    telluric : int
        telluric correction parameter
    img : array
        image of the swath

    Returns
    -------
    tell : array
        telluric correction
    """
    width, height = img.shape

    tel_lim = telluric if telluric > 5 and telluric < height / 2 else min(5, height / 3)
    tel = np.sum(img, axis=0)
    itel = np.arange(height)
    itel = itel[np.abs(itel - height / 2) >= tel_lim]
    tel = img[itel, :]
    sc = np.zeros(width)

    for itel in range(width):
        sc[itel] = np.ma.median(tel[itel])

    return sc


def calc_scatter_correction(scatter, index):
    """Calculate scatter correction
    by interpolating between values?

    Parameters
    ----------
    scatter : array of shape (degree_x, degree_y)
        2D polynomial coefficients of the background scatter
    index : tuple (array, array)
        indices of the swath within the overall image

    Returns
    -------
    scatter_correction : array of shape (swath_width, swath_height)
        correction for scattered light
    """

    # The indices in the image are switched
    y, x = index
    scatter_correction = np.polynomial.polynomial.polyval2d(x, y, scatter)
    return scatter_correction


def extract_spectrum(
    img,
    ycen,
    yrange,
    xrange,
    gain=1,
    readnoise=0,
    lambda_sf=0.1,
    lambda_sp=0,
    osample=1,
    swath_width=None,
    maxiter=20,
    reject_threshold=6,
    telluric=None,
    scatter=None,
    normalize=False,
    threshold=0,
    curvature=None,
    plot=False,
    plot_title=None,
    im_norm=None,
    im_ordr=None,
    out_spec=None,
    out_sunc=None,
    out_slitf=None,
    out_mask=None,
    progress=None,
    ord_num=0,
    preset_slitfunc=None,
    **kwargs,
):
    """
    Extract the spectrum of a single order from an image
    The order is split into several swathes of roughly swath_width length, which overlap half-half
    For each swath a spectrum and slitfunction are extracted
    overlapping sections are combined using linear weights (centrum is strongest, falling off to the edges)
    Here is the layout for the bins:

    ::

           1st swath    3rd swath    5th swath      ...
        /============|============|============|============|============|

                  2nd swath    4th swath    6th swath
               |------------|------------|------------|------------|
               |.....|
               overlap

               +     ******* 1
                +   *
                 + *
                  *            weights (+) previous swath, (*) current swath
                 * +
                *   +
               *     +++++++ 0

    Parameters
    ----------
    img : array[nrow, ncol]
        observation (or similar)
    ycen : array[ncol]
        order trace of the current order
    yrange : tuple(int, int)
        extraction width in pixles, below and above
    xrange : tuple(int, int)
        columns range to extract (low, high)
    gain : float, optional
        adu to electron, amplifier gain (default: 1)
    readnoise : float, optional
        read out noise factor (default: 0)
    lambda_sf : float, optional
        slit function smoothing parameter, usually very small (default: 0.1)
    lambda_sp : int, optional
        spectrum smoothing parameter, usually very small (default: 0)
    osample : int, optional
        oversampling factor, i.e. how many subpixels to create per pixel (default: 1, i.e. no oversampling)
    swath_width : int, optional
        swath width suggestion, actual width depends also on ncol, see make_bins (default: None, which will determine the width based on the order tracing)
    telluric : {float, None}, optional
        telluric correction factor (default: None, i.e. no telluric correction)
    scatter : {array, None}, optional
        background scatter as 2d polynomial coefficients (default: None, no correction)
    normalize : bool, optional
        whether to create a normalized image. If true, im_norm and im_ordr are used as output (default: False)
    threshold : int, optional
        threshold for normalization (default: 0)
    curvature : array[ncol, n_coeffs], optional
        Slit curvature polynomial coefficients for this trace (default: None, i.e. vertical extraction)
    plot : bool, optional
        wether to plot the progress, plotting will slow down the procedure significantly (default: False)
    ord_num : int, optional
        current order number, just for plotting (default: 0)
    im_norm : array[nrow, ncol], optional
        normalized image, only output if normalize is True (default: None)
    im_ordr : array[nrow, ncol], optional
        image of the order blaze, only output if normalize is True (default: None)

    Returns
    -------
    spec : array[ncol]
        extracted spectrum
    slitf : array[nslitf]
        extracted slitfunction
    mask : array[ncol]
        mask of the column range to use in the spectrum
    unc : array[ncol]
        uncertainty on the spectrum
    """

    _, ncol = img.shape
    ylow, yhigh = yrange
    xlow, xhigh = xrange
    nslitf = osample * (ylow + yhigh + 2) + 1

    # Validate preset_slitfunc size before extraction
    if preset_slitfunc is not None and len(preset_slitfunc) != nslitf:
        raise ValueError(
            f"preset_slitfunc size mismatch: got {len(preset_slitfunc)} elements, "
            f"expected {nslitf} for osample={osample}, yrange=({ylow}, {yhigh}). "
            f"Ensure norm_flat and extraction use the same extraction_height and osample."
        )

    ycen_int = np.floor(ycen).astype(int)

    spec = np.zeros(ncol) if out_spec is None else out_spec
    sunc = np.zeros(ncol) if out_sunc is None else out_sunc
    mask = np.full(ncol, False) if out_mask is None else out_mask
    slitf = np.zeros(nslitf) if out_slitf is None else out_slitf

    nbin, bins_start, bins_end = make_bins(swath_width, xlow, xhigh, ycen)
    nswath = 2 * nbin - 1
    swath = Swath(nswath)
    margin = np.zeros((nswath, 2), int)

    if normalize:
        norm_img = [None] * nswath
        norm_model = [None] * nswath

    # Perform slit decomposition within each swath stepping through the order with
    # half swath width. Spectra for each decomposition are combined with linear weights.
    with tqdm(
        enumerate(zip(bins_start, bins_end, strict=False)),
        total=len(bins_start),
        leave=False,
        desc="Swath",
    ) as t:
        for ihalf, (ibeg, iend) in t:
            logger.debug("Extracting Swath %i, Columns: %i - %i", ihalf, ibeg, iend)

            # Cut out swath from image
            index = make_index(ycen_int - ylow, ycen_int + yhigh, ibeg, iend)
            swath_img = img[index]
            # Convert ycen to swath-relative coordinates
            # The swath is cut from ycen_int - ylow, so within the swath:
            # trace center = ylow + fractional_part(ycen)
            swath_ycen_abs = ycen[ibeg:iend]
            swath_ycen = ylow + (swath_ycen_abs - np.floor(swath_ycen_abs))

            # Corrections
            # TODO: what is it even supposed to do?
            if telluric is not None:  # pragma: no cover
                telluric_correction = calc_telluric_correction(telluric, swath_img)
            else:
                telluric_correction = 0

            if scatter is not None:
                scatter_correction = calc_scatter_correction(scatter, index)
            else:
                scatter_correction = 0

            swath_img -= scatter_correction + telluric_correction

            # Do Slitfunction extraction
            swath_ncols = iend - ibeg
            swath_curv = curvature[ibeg:iend] if curvature is not None else None
            input_mask = np.ma.getmaskarray(swath_img).copy()

            if USE_CHARSLIT:
                slitcurve = _ensure_slitcurve(swath_curv, swath_ncols)
                slitdeltas = np.zeros(swath_img.shape[0], dtype=np.float64)
                swath[ihalf] = _slitdec_charslit(
                    swath_img,
                    swath_ycen,
                    slitcurve,
                    slitdeltas,
                    lambda_sp=lambda_sp,
                    lambda_sf=lambda_sf,
                    osample=osample,
                    yrange=yrange,
                    maxiter=maxiter,
                    gain=gain,
                    reject_threshold=reject_threshold,
                    preset_slitfunc=preset_slitfunc,
                )
            else:
                # CFFI backend only supports degree <= 2
                if swath_curv is not None and swath_curv.shape[1] > 3:
                    raise ValueError(
                        "curve_degree > 2 requires charslit. "
                        "Set PYREDUCE_USE_CHARSLIT=1 to enable."
                    )
                swath[ihalf] = _slitdec_cffi(
                    swath_img,
                    swath_ycen,
                    swath_curv,
                    lambda_sp=lambda_sp,
                    lambda_sf=lambda_sf,
                    osample=osample,
                    yrange=yrange,
                    maxiter=maxiter,
                    gain=gain,
                    reject_threshold=reject_threshold,
                    preset_slitfunc=preset_slitfunc,
                )
            t.set_postfix(chi=f"{swath[ihalf][5][1]:1.2f}")

            if normalize:
                # Save image and model for later
                # Use np.divide to avoid divisions by zero
                where = swath.model[ihalf] > threshold / gain
                norm_img[ihalf] = np.ones_like(swath.model[ihalf])
                np.divide(
                    np.abs(swath_img),
                    swath.model[ihalf],
                    where=where,
                    out=norm_img[ihalf],
                )
                norm_model[ihalf] = swath.model[ihalf]

            if (
                plot >= 2
                and not np.all(np.isnan(swath_img))
                and util.is_interactive_plot_mode()
            ):  # pragma: no cover
                if progress is None:
                    progress = ProgressPlot(
                        swath_img.shape[0], swath_img.shape[1], nslitf, title=plot_title
                    )
                progress.plot(
                    swath_img,
                    swath.spec[ihalf],
                    swath.slitf[ihalf],
                    swath.model[ihalf],
                    swath_ycen,
                    input_mask,
                    swath.mask[ihalf],
                    ord_num,
                    ibeg,
                    iend,
                    swath.unc[ihalf],
                    swath.info[ihalf],
                    ihalf,
                )

    # Remove points at the border of the each swath, if order has curvature
    # as those pixels have bad information
    for i in range(nswath):
        margin[i, :] = int(swath.info[i][4]) + 1

    # Weight for combining swaths
    weight = [np.ones(bins_end[i] - bins_start[i]) for i in range(nswath)]
    weight[0][: margin[0, 0]] = 0
    weight[-1][len(weight[-1]) - margin[-1, 1] :] = 0
    for i, j in zip(range(0, nswath - 1), range(1, nswath), strict=False):
        width = bins_end[i] - bins_start[i]
        overlap = bins_end[i] - bins_start[j]

        # Start and end indices for the two swaths
        start_i = width - overlap + margin[j, 0]
        end_i = width - margin[i, 1]

        start_j = margin[j, 0]
        end_j = overlap - margin[i, 1]

        # Weights for one overlap from 0 to 1, but do not include those values (whats the point?)
        triangle = np.linspace(0, 1, overlap + 1, endpoint=False)[1:]
        # Cut away the margins at the corners
        triangle = triangle[margin[j, 0] : len(triangle) - margin[i, 1]]

        # Set values
        weight[i][start_i:end_i] = 1 - triangle
        weight[j][start_j:end_j] = triangle

        # Don't use the pixels at the egdes (due to curvature)
        weight[i][end_i:] = 0
        weight[j][:start_j] = 0

    # Update column range
    xrange[0] += margin[0, 0]
    xrange[1] -= margin[-1, 1]
    mask[: xrange[0]] = True
    mask[xrange[1] :] = True

    # Apply weights
    for i, (ibeg, iend) in enumerate(zip(bins_start, bins_end, strict=False)):
        spec[ibeg:iend] += swath.spec[i] * weight[i]
        sunc[ibeg:iend] += swath.unc[i] * weight[i]

    if normalize:
        for i, (ibeg, iend) in enumerate(zip(bins_start, bins_end, strict=False)):
            index = make_index(ycen_int - ylow, ycen_int + yhigh, ibeg, iend)
            im_norm[index] += norm_img[i] * weight[i]
            im_ordr[index] += norm_model[i] * weight[i]

    slitf[:] = np.mean(swath.slitf, axis=0)
    sunc[:] = np.sqrt(sunc**2 + (readnoise / gain) ** 2)
    return spec, slitf, mask, sunc


def model(spec, slitf):
    return spec[None, :] * slitf[:, None]


def get_y_scale(ycen, xrange, extraction_height, nrow):
    """Calculate the y limits of the order for C extraction code.

    Parameters
    ----------
    ycen : array[ncol]
        order trace
    xrange : tuple(int, int)
        column range
    extraction_height : int
        extraction full height in pixels
    nrow : int
        number of rows in the image, defines upper edge

    Returns
    -------
    y_low, y_high : int, int
        lower and upper y bound for extraction (pixels below/above trace)
        These satisfy: y_low + y_high + 1 = extraction_height
    """
    ycen = ycen[xrange[0] : xrange[1]]
    half = extraction_height // 2

    ymin = ycen - half
    ymin = np.floor(ymin)
    if min(ymin) < 0:
        ymin = ymin - min(ymin)  # help for orders at edge
    if max(ymin) >= nrow:
        ymin = ymin - max(ymin) + nrow - 1  # helps at edge

    ymax = ymin + extraction_height - 1
    if max(ymax) >= nrow:
        ymax = ymax - max(ymax) + nrow - 1  # helps at edge
        ymin = ymax - extraction_height + 1

    # Define a fixed height area containing one spectral order
    y_lower_lim = int(np.min(ycen - ymin))  # Pixels below center line
    y_upper_lim = int(np.min(ymax - ycen))  # Pixels above center line

    return y_lower_lim, y_upper_lim


def optimal_extraction(
    img,
    traces,
    extraction_height,
    column_range,
    curvature=None,
    plot=False,
    plot_title=None,
    **kwargs,
):
    """Use optimal extraction to get spectra

    This functions just loops over the traces, the actual work is done in extract_spectrum

    Parameters
    ----------
    img : array[nrow, ncol]
        image to extract
    traces : array[ntrace, degree]
        trace polynomial coefficients
    extraction_height : array[ntrace]
        extraction full height in pixels
    column_range : array[ntrace, 2]
        column range to use
    curvature : array[ntrace, ncol, n_coeffs] or None
        Slit curvature polynomial coefficients (default: None for vertical extraction)
    **kwargs
        other parameters for the extraction (see extract_spectrum)

    Returns
    -------
    spectrum : array[ntrace, ncol]
        extracted spectrum
    slitfunction : array[ntrace, nslitf]
        recovered slitfunction
    uncertainties: array[ntrace, ncol]
        uncertainties on the spectrum
    """

    logger.info("Using optimal extraction to produce spectrum")

    nrow, ncol = img.shape
    ntrace = len(traces)

    spectrum = np.zeros((ntrace, ncol))
    uncertainties = np.zeros((ntrace, ncol))
    slitfunction = [None for _ in range(ntrace)]

    # Handle preset_slitfunc (list of per-trace slitfuncs)
    preset_slitfunc = kwargs.pop("preset_slitfunc", None)

    # Add mask as defined by column ranges
    mask = np.full((ntrace, ncol), True)
    for i in range(ntrace):
        mask[i, column_range[i, 0] : column_range[i, 1]] = False
    spectrum = np.ma.array(spectrum, mask=mask)
    uncertainties = np.ma.array(uncertainties, mask=mask)

    ix = np.arange(ncol)
    if plot >= 2 and util.is_interactive_plot_mode():  # pragma: no cover
        ncol_swath = kwargs.get("swath_width", img.shape[1] // 400)
        nrow_swath = np.max(extraction_height)
        nslitf_swath = (nrow_swath + 2) * kwargs.get("osample", 1) + 1
        progress = ProgressPlot(nrow_swath, ncol_swath, nslitf_swath, title=plot_title)
    else:
        progress = None

    for i in tqdm(range(ntrace), desc="Trace"):
        logger.debug("Extracting trace %i out of %i", i + 1, ntrace)

        # Define a fixed height area containing one trace
        ycen = np.polyval(traces[i], ix)
        yrange = get_y_scale(ycen, column_range[i], extraction_height[i], nrow)

        osample = kwargs.get("osample", 1)
        slitfunction[i] = np.zeros(osample * (sum(yrange) + 2) + 1)

        # Return values are set by reference, as the out parameters
        # Also column_range is adjusted depending on the curvature
        # This is to avoid large chunks of memory of essentially duplicates
        order_slitfunc = None
        if preset_slitfunc is not None and i < len(preset_slitfunc):
            order_slitfunc = preset_slitfunc[i]
        trace_curv = curvature[i] if curvature is not None else None
        extract_spectrum(
            img,
            ycen,
            yrange,
            column_range[i],
            curvature=trace_curv,
            out_spec=spectrum[i],
            out_sunc=uncertainties[i],
            out_slitf=slitfunction[i],
            out_mask=mask[i],
            progress=progress,
            ord_num=i + 1,
            plot=plot,
            plot_title=plot_title,
            preset_slitfunc=order_slitfunc,
            **kwargs,
        )

    if plot >= 2 and progress is not None:  # pragma: no cover
        progress.close()

    if plot:  # pragma: no cover
        plot_comparison(
            img,
            traces,
            spectrum,
            slitfunction,
            extraction_height,
            column_range,
            title=plot_title,
        )

    return spectrum, slitfunction, uncertainties


def correct_for_curvature(img_order, curvature, xwd, inverse=False):
    """Correct image for slit curvature by interpolation.

    Parameters
    ----------
    img_order : array[nrow, ncol]
        Image swath to correct
    curvature : array[ncol, n_coeffs]
        Curvature coefficients [c0, c1, c2, ...] where dx = c1*y + c2*y^2 + ...
    xwd : int
        Extraction full height in pixels
    inverse : bool
        If True, apply inverse correction (for model reapplication)

    Returns
    -------
    img_order : array
        Corrected image
    """
    mask = ~np.ma.getmaskarray(img_order)
    sign = -1 if inverse else 1
    half = xwd // 2

    xt = np.arange(img_order.shape[1])
    for y, yt in zip(range(xwd), range(-half, xwd - half), strict=False):
        # Compute displacement: dx = c1*y + c2*y^2 + c3*y^3 + ...
        dx = np.zeros(img_order.shape[1])
        for k in range(1, curvature.shape[1]):
            dx += curvature[:, k] * (yt**k)
        xi = xt + sign * dx
        img_order[y] = np.interp(
            xi, xt[mask[y]], img_order[y][mask[y]], left=0, right=0
        )

    xt = np.arange(img_order.shape[0])
    for x in range(img_order.shape[1]):
        img_order[:, x] = np.interp(
            xt, xt[mask[:, x]], img_order[:, x][mask[:, x]], left=0, right=0
        )

    return img_order


def model_image(img, xwd, curvature):
    """Create model image from curvature-corrected data."""
    img = correct_for_curvature(img, curvature, xwd)
    # Find slitfunction using the median to avoid outliers
    slitf = np.ma.median(img, axis=1)
    slitf /= np.ma.sum(slitf)
    # Use the slitfunction to find spectrum
    spec = np.ma.median(img / slitf[:, None], axis=0)
    # Create model from slitfunction and spectrum
    model = spec[None, :] * slitf[:, None]
    # Reapply curvature to the model (inverse)
    model = correct_for_curvature(model, curvature, xwd, inverse=True)
    return model, spec, slitf


def simple_extraction(
    img,
    traces,
    extraction_height,
    column_range,
    gain=1,
    readnoise=0,
    dark=0,
    plot=False,
    plot_title=None,
    curvature=None,
    collapse_function="median",
    **kwargs,
):
    """Use simple extraction to get a spectrum
    Simple extraction takes the sum/mean/median orthogonal to the trace for extraction_height pixels

    This extraction makes a few rough assumptions and does not provide the most accurate results,
    but rather a good approximation

    Parameters
    ----------
    img : array[nrow, ncol]
        image to extract
    traces : array[ntrace, degree]
        trace polynomial coefficients
    extraction_height : array[ntrace]
        extraction full height in pixels
    column_range : array[ntrace, 2]
        column range to use
    gain : float, optional
        adu to electron, amplifier gain (default: 1)
    readnoise : float, optional
        read out noise (default: 0)
    dark : float, optional
        dark current noise (default: 0)
    plot : bool, optional
        wether to plot the results (default: False)

    Returns
    -------
    spectrum : array[ntrace, ncol]
        extracted spectrum
    uncertainties : array[ntrace, ncol]
        uncertainties on extracted spectrum
    """

    logger.info("Using simple extraction to produce spectrum")
    _, ncol = img.shape
    ntrace, _ = traces.shape

    spectrum = np.zeros((ntrace, ncol))
    uncertainties = np.zeros((ntrace, ncol))

    # Add mask as defined by column ranges
    mask = np.full((ntrace, ncol), True)
    for i in range(ntrace):
        mask[i, column_range[i, 0] : column_range[i, 1]] = False
    spectrum = np.ma.array(spectrum, mask=mask)
    uncertainties = np.ma.array(uncertainties, mask=mask)

    x = np.arange(ncol)

    for i in tqdm(range(ntrace), desc="Trace"):
        logger.debug("Extracting trace %i out of %i", i + 1, ntrace)

        x_left_lim = column_range[i, 0]
        x_right_lim = column_range[i, 1]

        # Rectify the image, i.e. remove the shape of the trace
        # Then the center of the trace is within one pixel variations
        ycen = np.polyval(traces[i], x).astype(int)
        half = extraction_height[i] // 2
        yb = ycen - half
        yt = yb + extraction_height[i] - 1
        index = make_index(yb, yt, x_left_lim, x_right_lim)
        img_trace = img[index]

        # Correct for curvature
        # For each row of the rectified trace, interpolate onto the shifted row
        # Masked pixels are set to 0, similar to the summation
        if curvature is not None:
            trace_curv = curvature[i, x_left_lim:x_right_lim]
            img_trace = correct_for_curvature(
                img_trace,
                trace_curv,
                extraction_height[i],
            )

        # Sum over the prepared image
        if collapse_function == "sum":
            arc = np.ma.sum(img_trace, axis=0)
        elif collapse_function == "mean":
            arc = np.ma.mean(img_trace, axis=0) * img_trace.shape[0]
        elif collapse_function == "median":
            arc = np.ma.median(img_trace, axis=0) * img_trace.shape[0]
        else:
            raise ValueError(
                f"Could not determine the arc method, expected one of ('sum', 'mean', 'median'), but got {collapse_function}"
            )

        # Store results
        spectrum[i, x_left_lim:x_right_lim] = arc
        uncertainties[i, x_left_lim:x_right_lim] = (
            np.sqrt(np.abs(arc * gain + dark + readnoise**2)) / gain
        )

    if plot:  # pragma: no cover
        plot_comparison(
            img,
            traces,
            spectrum,
            None,
            extraction_height,
            column_range,
            title=plot_title,
        )

    return spectrum, uncertainties


def plot_comparison(
    original, traces, spectrum, slitf, extraction_height, column_range, title=None
):  # pragma: no cover
    plt.figure()
    nrow, ncol = original.shape
    ntrace = len(traces)
    output = np.zeros((np.sum(extraction_height) + ntrace, ncol))
    pos = [0]
    x = np.arange(ncol)
    for i in range(ntrace):
        ycen = np.polyval(traces[i], x)
        half = extraction_height[i] // 2
        yb = ycen - half
        yt = yb + extraction_height[i] - 1
        xl, xr = column_range[i]
        index = make_index(yb, yt, xl, xr)
        yl = pos[i]
        yr = pos[i] + index[0].shape[0]
        output[yl:yr, xl:xr] = original[index]

        vmin, vmax = np.percentile(output[yl:yr, xl:xr], (5, 95))
        output[yl:yr, xl:xr] = np.clip(output[yl:yr, xl:xr], vmin, vmax)
        output[yl:yr, xl:xr] -= vmin
        output[yl:yr, xl:xr] /= vmax - vmin

        pos += [yr]

    plt.imshow(output, origin="lower", aspect="auto")

    for i in range(ntrace):
        try:
            tmp = spectrum[i, column_range[i, 0] : column_range[i, 1]]
            # if len(tmp)
            vmin = np.min(tmp[tmp != 0])
            tmp = np.copy(spectrum[i])
            tmp[tmp != 0] -= vmin
            np.log(tmp, out=tmp, where=tmp > 0)
            tmp = tmp / np.max(tmp) * 0.9 * (pos[i + 1] - pos[i])
            tmp += pos[i]
            tmp[tmp < pos[i]] = pos[i]
            plt.plot(x, tmp, "r")
        except:
            pass

    locs = np.sum(extraction_height, axis=1) + 1
    locs = np.array([0, *np.cumsum(locs)[:-1]])
    locs[:-1] += (np.diff(locs) * 0.5).astype(int)
    locs[-1] += ((output.shape[0] - locs[-1]) * 0.5).astype(int)
    plt.yticks(locs, range(len(locs)))

    plot_title = "Extracted Spectrum vs. Rectified Image"
    if title is not None:
        plot_title = f"{title}\n{plot_title}"
    plt.title(plot_title)
    plt.xlabel("x [pixel]")
    plt.ylabel("trace")
    util.show_or_save("extract_rectify")


def extract(
    img,
    traces,
    column_range=None,
    trace_range=None,
    extraction_height=0.5,
    extraction_type="optimal",
    curvature=None,
    **kwargs,
):
    """
    Extract the spectrum from an image

    Parameters
    ----------
    img : array[nrow, ncol](float)
        observation to extract
    traces : array[ntrace, degree](float)
        polynomial coefficients of the trace positions
    column_range : array[ntrace, 2](int), optional
        range of pixels to use for each trace (default: use all)
    trace_range : array[2](int), optional
        range of traces to extract, traces have to be consecutive (default: use all)
    extraction_height : float, optional
        Total extraction height. Values below 3 are fractions of trace spacing, values above are pixels. Split evenly above/below trace. (default: 1.0)
    extraction_type : {"optimal", "simple", "normalize"}, optional
        which extraction algorithm to use, "optimal" uses optimal extraction, "simple" uses simple sum/median extraction, and "normalize" also uses optimal extraction, but returns the normalized image (default: "optimal")
    curvature : array[ntrace, ncol, n_coeffs], optional
        Slit curvature polynomial coefficients (default: None for vertical extraction)
    polarization : bool, optional
        if true, pairs of traces are considered to belong to the same order, but different polarization. Only affects the scatter (default: False)
    **kwargs, optional
        parameters for extraction functions

    Returns
    -------
    spec : array[ntrace, ncol](float)
        extracted spectrum for each trace
    uncertainties : array[ntrace, ncol](float)
        uncertainties on the spectrum

    if extraction_type == "normalize" instead return

    im_norm : array[nrow, ncol](float)
        normalized image
    im_ordr : array[nrow, ncol](float)
        image with just the traces
    blaze : array[ntrace, ncol](float)
        extracted spectrum (equals blaze if img was the flat field)
    """

    nrow, ncol = img.shape
    ntrace, _ = traces.shape
    if trace_range is None:
        trace_range = (0, ntrace)

    # Fix the input parameters
    extraction_height, column_range, traces = fix_parameters(
        extraction_height, column_range, traces, nrow, ncol, ntrace
    )
    # Limit traces (and related properties) to traces in range
    ntrace = trace_range[1] - trace_range[0]
    traces = traces[trace_range[0] : trace_range[1]]
    column_range = column_range[trace_range[0] : trace_range[1]]
    extraction_height = extraction_height[trace_range[0] : trace_range[1]]
    if curvature is not None:
        curvature = curvature[trace_range[0] : trace_range[1]]

    if extraction_type == "optimal":
        # the "normal" case, except for wavelength calibration files
        spectrum, slitfunction, uncertainties = optimal_extraction(
            img,
            traces,
            extraction_height,
            column_range,
            curvature=curvature,
            **kwargs,
        )
    elif extraction_type == "normalize":
        # TODO
        # Prepare normalized flat field image if necessary
        # These will be passed and "returned" by reference
        # I dont like it, but it works for now
        im_norm = np.zeros_like(img)
        im_ordr = np.zeros_like(img)

        blaze, slitfunction, _ = optimal_extraction(
            img,
            traces,
            extraction_height,
            column_range,
            curvature=curvature,
            normalize=True,
            im_norm=im_norm,
            im_ordr=im_ordr,
            **kwargs,
        )
        threshold_lower = kwargs.get("threshold_lower", 0)
        im_norm[im_norm <= threshold_lower] = 1
        im_ordr[im_ordr <= threshold_lower] = 1
        return im_norm, im_ordr, blaze, slitfunction, column_range
    elif extraction_type in ("simple", "arc"):  # "arc" for backwards compatibility
        spectrum, uncertainties = simple_extraction(
            img,
            traces,
            extraction_height,
            column_range,
            curvature=curvature,
            **kwargs,
        )
        slitfunction = None
    else:
        raise ValueError(
            f"Parameter 'extraction_type' not understood. Expected 'optimal', 'normalize', or 'simple' but got {extraction_type}."
        )

    return spectrum, uncertainties, slitfunction, column_range
