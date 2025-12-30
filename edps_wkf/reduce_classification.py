"""Classification rules for raw and product files.

CRIRES+ specific patterns for now, will be generalized with adapters later.
"""

from edps import classification_rule

from . import reduce_keywords as kwd


def _is_bias(f):
    """CRIRES+ uses DARK for bias frames."""
    dpr_type = f[kwd.dpr_type]
    return dpr_type is not None and str(dpr_type).upper() == "DARK"


def _is_flat(f):
    dpr_type = f[kwd.dpr_type]
    return dpr_type is not None and str(dpr_type).upper() == "FLAT"


def _is_wave(f):
    dpr_type = f[kwd.dpr_type]
    return dpr_type is not None and "WAVE" in str(dpr_type).upper()


def _is_science(f):
    dpr_catg = f[kwd.dpr_catg]
    return dpr_catg is not None and str(dpr_catg).upper() == "SCIENCE"


# Raw frame classification rules
bias_class = classification_rule("BIAS", _is_bias)
flat_class = classification_rule("FLAT", _is_flat)
wave_class = classification_rule("WAVE", _is_wave)
science_class = classification_rule("SCIENCE", _is_science)

# Product classification rules (by PRO.CATG header)
master_bias_class = classification_rule("MASTER_BIAS", {kwd.pro_catg: "MASTER_BIAS"})
master_flat_class = classification_rule("MASTER_FLAT", {kwd.pro_catg: "MASTER_FLAT"})
orders_class = classification_rule("ORDERS", {kwd.pro_catg: "ORDERS"})
wave_solution_class = classification_rule(
    "WAVE_SOLUTION", {kwd.pro_catg: "WAVE_SOLUTION"}
)
