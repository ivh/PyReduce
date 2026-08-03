"""Reduction step classes, one module per step family.

The Pipeline class in :mod:`pyreduce.pipeline` wires these together;
``pyreduce.reduce`` re-exports everything here for backwards compatibility.
"""

from .base import (
    CalibrationStep,
    ExtractionStep,
    Step,
    wavelengths_from_traces,
)
from .calibration import (
    BackgroundScatter,
    Bias,
    Flat,
    Mask,
    NormalizeFlatField,
)
from .extraction import (
    RectifyImage,
    ScienceExtraction,
)
from .output import (
    ContinuumNormalization,
    Finalize,
)
from .trace import (
    SlitCurvatureDetermination,
    Trace,
)
from .wavecal import (
    LaserFrequencyCombFinalize,
    LaserFrequencyCombMaster,
    WavelengthCalibrationFinalize,
    WavelengthCalibrationInitialize,
    WavelengthCalibrationMaster,
)

__all__ = [
    "BackgroundScatter",
    "Bias",
    "CalibrationStep",
    "ContinuumNormalization",
    "ExtractionStep",
    "Finalize",
    "Flat",
    "LaserFrequencyCombFinalize",
    "LaserFrequencyCombMaster",
    "Mask",
    "NormalizeFlatField",
    "RectifyImage",
    "ScienceExtraction",
    "SlitCurvatureDetermination",
    "Step",
    "Trace",
    "WavelengthCalibrationFinalize",
    "WavelengthCalibrationInitialize",
    "WavelengthCalibrationMaster",
    "wavelengths_from_traces",
]
