"""
MOSAIC instrument - ELT multi-object spectrograph.

NIR channel: 630 fibers in 90 bundles, 4096x4096 H4RG detector.
VIS channel: 966 fibers in 138 bundles, 4 quadrants of a 12788x12394 mosaic.
Each VIS quadrant sees one half of the fiber slit (top or bottom).

The raw FITS files carry a FIBRE_TABLE extension that pairs each fiber with
its bundle id and in-bundle index. MOSAIC.assign_bundles() consumes this
to set t.bundle / t.fiber_idx without needing a bundle_centers YAML.
"""

import logging

from astropy.io import fits

from ..common import Instrument

logger = logging.getLogger(__name__)


class MOSAIC(Instrument):
    # VIS detector quadrant boundaries (from gap analysis)
    # Full VIS frame: 12788 (height) x 12394 (width) pixels
    # Horizontal gap: rows 6144-6644, Vertical gap: cols 6144-6250
    # Active region: rows 2038-10734
    #
    # slit_half tells assign_bundles() which contiguous block of live
    # FIBRE_TABLE rows applies to this quadrant:
    #   "top"    -> the lowest INDEX values (top of slit in detector y)
    #   "bottom" -> the highest INDEX values
    VIS_QUADRANTS = {
        "VIS1": {
            "xlo": 0,
            "xhi": 6144,
            "ylo": 1780,
            "yhi": 6000,
            "slit_half": "bottom",
        },
        "VIS2": {
            "xlo": 6249,
            "xhi": 12393,
            "ylo": 1810,
            "yhi": 6005,
            "slit_half": "bottom",
        },
        "VIS3": {
            "xlo": 0,
            "xhi": 6144,
            "ylo": 6775,
            "yhi": 10990,
            "slit_half": "top",
        },
        "VIS4": {
            "xlo": 6249,
            "xhi": 12393,
            "ylo": 6644,
            "yhi": 10945,
            "slit_half": "top",
        },
    }

    def add_header_info(self, header, channel, **kwargs):
        """Override to handle VIS quadrant extraction."""
        header = super().add_header_info(header, channel, **kwargs)

        channel_upper = channel.upper()
        if channel_upper in self.VIS_QUADRANTS:
            q = self.VIS_QUADRANTS[channel_upper]
            header["e_xlo"] = q["xlo"]
            header["e_xhi"] = q["xhi"]
            header["e_ylo"] = q["ylo"]
            header["e_yhi"] = q["yhi"]

        return header

    def assign_bundles(self, traces, files, header, channel):
        """Pair detected traces with FIBRE_TABLE rows from the raw FITS.

        FIBRE_TABLE INDEX order matches detected y order within each VIS
        quadrant (verified empirically for the May-2026 sim batch). The
        contiguous block of live rows assigned to this quadrant is the
        first N for slit_half="top" and the last N for slit_half="bottom",
        where N is the count of detected traces.

        NIR channels see all 630 fibers (no half-split), so the entire live
        FIBRE_TABLE is used.

        Returns a ``{bundle_id: y_center}`` dict where each y_center is the
        y of the geometric middle fiber (IFB == bundle_size // 2) when
        detected, falling back to the average of the symmetric pair around
        it, or the mean of all detected fibers as a last resort. Bundles
        within MOSAIC have a non-uniform internal spacing (a gap between
        the two slitlet halves), so mean-of-detected is not the geometric
        center; this routine returns the right reference for center_weight.

        Bails (raises) on count mismatches so a silent off-by-one can't
        propagate.
        """
        if not files:
            return None
        try:
            with fits.open(files[0]) as h:
                if "FIBRE_TABLE" not in h:
                    return None
                fibre = h["FIBRE_TABLE"].data.copy()
        except Exception as exc:
            logger.warning("Could not read FIBRE_TABLE from %s: %s", files[0], exc)
            return None

        live = fibre[fibre["HEALTH"] == 1]
        slit_half = self.VIS_QUADRANTS.get(channel.upper(), {}).get("slit_half")
        n_detected = len(traces)

        if slit_half == "top":
            subset = live[:n_detected]
        elif slit_half == "bottom":
            subset = live[-n_detected:]
        else:
            # NIR / single-region: use all live fibers
            subset = live

        if len(subset) != n_detected:
            raise ValueError(
                f"MOSAIC.assign_bundles: detected {n_detected} traces but "
                f"FIBRE_TABLE provided {len(subset)} live rows for channel "
                f"{channel!r} (slit_half={slit_half!r}). Refusing to assign "
                "to avoid silent off-by-one."
            )

        if not traces:
            return None

        # Match by y order: sort traces by y at column midpoint, pair 1:1.
        x_mid = sum(traces[0].column_range) / 2
        order = sorted(range(n_detected), key=lambda i: traces[i].y_at_x(x_mid))
        for rank, i in enumerate(order):
            row = subset[rank]
            traces[i].bundle = int(row["BUNDLE"])
            traces[i].fiber_idx = int(row["IFB"]) + 1  # 1-based in PyReduce
            # MOSAIC is single-order: clear the per-trace m that mark_orders
            # assigned sequentially, so group_fibers collects by bundle alone.
            traces[i].m = None

        # Build bundle_centers using IFB instead of a biased mean of y.
        # Group traces by bundle, then pick the fiber whose IFB is closest
        # to the geometric middle slot (size // 2). When two slots are tied
        # equidistant (center missing, both flanks present), average them.
        from collections import defaultdict

        bundle_size = getattr(self.config, "fibers", None)
        bundle_size = (
            bundle_size.bundles.size
            if bundle_size is not None and bundle_size.bundles is not None
            else 7
        )
        middle_slot = bundle_size // 2  # 0-based IFB

        per_bundle: dict[int, list] = defaultdict(list)
        for t in traces:
            if t.bundle is None or t.fiber_idx is None:
                continue
            per_bundle[int(t.bundle)].append(t)

        bundle_centers: dict[int, float] = {}
        for b, members in per_bundle.items():
            # delta = distance to geometric middle slot (in IFB units)
            keyed = [(abs((t.fiber_idx - 1) - middle_slot), t) for t in members]
            keyed.sort(key=lambda kv: kv[0])
            best_delta = keyed[0][0]
            chosen = [t for d, t in keyed if d == best_delta]
            ys = [t.y_at_x(sum(t.column_range) / 2) for t in chosen]
            bundle_centers[b] = float(sum(ys) / len(ys))

        logger.info(
            "MOSAIC.assign_bundles: paired %d traces from FIBRE_TABLE "
            "(channel=%s, slit_half=%s, %d unique bundles)",
            n_detected,
            channel,
            slit_half,
            len(bundle_centers),
        )
        return bundle_centers
