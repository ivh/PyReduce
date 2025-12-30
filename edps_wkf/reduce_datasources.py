"""Data source definitions with calibration association rules."""

from edps import data_source
from edps.generator.time_range import ONE_DAY, ONE_WEEK, UNLIMITED

from .reduce_classification import bias_class, flat_class, science_class, wave_class

raw_bias = (
    data_source("BIAS")
    .with_classification_rule(bias_class)
    .with_grouping_keywords(["mjd-obs"])
    .with_match_keywords(["instrume"], time_range=ONE_WEEK, level=0)
    .with_match_keywords(["instrume"], time_range=UNLIMITED, level=1)
    .build()
)

raw_flat = (
    data_source("FLAT")
    .with_classification_rule(flat_class)
    .with_grouping_keywords(["mjd-obs"])
    .with_match_keywords(["instrume"], time_range=ONE_DAY, level=0)
    .with_match_keywords(["instrume"], time_range=ONE_WEEK, level=1)
    .build()
)

raw_wave = (
    data_source("WAVE")
    .with_classification_rule(wave_class)
    .with_grouping_keywords(["mjd-obs"])
    .with_match_keywords(["instrume"], time_range=ONE_DAY, level=0)
    .build()
)

raw_science = (
    data_source("SCIENCE")
    .with_classification_rule(science_class)
    .with_grouping_keywords(["mjd-obs", "object"])
    .build()
)
