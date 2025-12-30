"""PyReduce EDPS workflow definition.

Usage:
    edps -w edps_wkf.reduce_wkf -i /path/to/data -c  # classify only
    edps -w edps_wkf.reduce_wkf -i /path/to/data -t bias -o /tmp/out
"""

from edps import QC1_CALIB, task

from .reduce_classification import master_bias_class
from .reduce_datasources import raw_bias, raw_flat

__title__ = "PyReduce Generic Workflow"

# Task: Create master bias
bias_task = (
    task("bias")
    .with_recipe("reduce_bias")
    .with_main_input(raw_bias)
    .with_meta_targets([QC1_CALIB])
    .build()
)

# Task: Create master flat
flat_task = (
    task("flat")
    .with_recipe("reduce_flat")
    .with_main_input(raw_flat)
    .with_associated_input(bias_task, [master_bias_class])
    .with_meta_targets([QC1_CALIB])
    .build()
)
