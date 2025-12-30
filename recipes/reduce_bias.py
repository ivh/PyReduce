"""PyReduce bias recipe wrapper for EDPS/pyesorex."""

from typing import Any

import cpl.core
import cpl.ui


class ReduceBias(cpl.ui.PyRecipe):
    """Create master bias from raw bias/dark frames."""

    _name = "reduce_bias"
    _version = "0.1"
    _author = "PyReduce"
    _email = "thomas.marquart@astro.uu.se"
    _copyright = "GPL-3.0-or-later"
    _synopsis = "Create master bias from raw bias frames"
    _description = "Combines bias frames using PyReduce's bias algorithm"

    def __init__(self):
        self.parameters = cpl.ui.ParameterList([])

    def run(
        self, frameset: cpl.ui.FrameSet, settings: dict[str, Any]
    ) -> cpl.ui.FrameSet:
        from astropy.io import fits

        from pyreduce.combine_frames import combine_bias
        from pyreduce.instruments.instrument_info import load_instrument

        # Get input files
        bias_files = [f.file for f in frameset if f.tag == "BIAS"]
        if not bias_files:
            raise ValueError("No BIAS frames provided")

        # Detect instrument from first file header
        with fits.open(bias_files[0]) as hdu:
            header = hdu[0].header
            inst_name = header.get("INSTRUME", "UNKNOWN")

        # Map header instrument name to PyReduce module name
        inst_map = {
            "CRIRES": "crires_plus",
        }
        module_name = inst_map.get(inst_name, inst_name)

        # Load PyReduce instrument
        instrument = load_instrument(module_name)

        # Get arm from header if instrument supports it
        arm = ""
        if hasattr(instrument, "get_arm_from_header"):
            with fits.open(bias_files[0]) as hdu:
                arm = instrument.get_arm_from_header(hdu[0].header)

        # Run PyReduce bias combination
        bias, bhead = combine_bias(
            bias_files,
            instrument,
            arm=arm,
            mask=None,
        )

        # Write output
        import numpy as np

        output_file = "MASTER_BIAS.fits"
        hdu = fits.PrimaryHDU(
            data=np.asarray(bias.data, dtype=np.float32), header=bhead
        )
        hdu.header["HIERARCH ESO PRO CATG"] = "MASTER_BIAS"
        hdu.header["HIERARCH ESO PRO TYPE"] = "REDUCED"
        hdu.writeto(output_file, overwrite=True)

        # Return output frameset
        output = cpl.ui.FrameSet()
        output.append(
            cpl.ui.Frame(
                file=output_file,
                tag="MASTER_BIAS",
                group=cpl.ui.Frame.FrameGroup.PRODUCT,
            )
        )
        return output
