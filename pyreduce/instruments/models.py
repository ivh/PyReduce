"""Pydantic models for instrument configuration validation.

These models provide type-safe validation for instrument configuration files.
Currently validates the flat YAML structure; will evolve toward the nested
structure described in REDESIGN.md.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class HeaderRef(BaseModel):
    """Reference to a FITS header keyword."""

    key: str

    model_config = ConfigDict(extra="forbid")


# Type for values that can be either a literal or a header reference
HeaderOrValue = float | int | str | HeaderRef | None


class FileClassification(BaseModel):
    """Keywords and patterns for file type classification."""

    kw_bias: str | None = None
    kw_flat: str | None = None
    kw_curvature: str | None = None
    kw_scatter: str | None = None
    kw_orders: str | None = None
    kw_wave: str | None = None
    kw_comb: str | None = None
    kw_spec: str | None = None

    id_bias: str | None = None
    id_flat: str | None = None
    id_curvature: str | None = None
    id_scatter: str | None = None
    id_orders: str | None = None
    id_wave: str | None = None
    id_comb: str | None = None
    id_spec: str | None = None

    model_config = ConfigDict(extra="allow")


class AmplifiersConfig(BaseModel):
    """Configuration for multi-amplifier readout assembly.

    Used when detector data is split across multiple FITS extensions,
    each with its own gain/readnoise and region mapping.
    """

    # Extension naming: template with {n} placeholder, e.g. "AMPLIFIER {n:02d}"
    extension_template: str
    # Number of amplifiers: header keyword or literal int
    count: str | int
    # Header keywords in each extension for calibration values
    gain: str = "GAIN"
    readnoise: str = "RDNOISE"
    # Header keywords for region mapping (IRAF section format)
    datasec: str = "DATASEC"
    detsec: str = "DETSEC"

    model_config = ConfigDict(extra="forbid")


class FiberGroupConfig(BaseModel):
    """Configuration for a named fiber group.

    Defines a range of physical fibers that form a logical group (e.g., "A", "cal", "B").
    The merged trace can be the average, center, or specific fiber(s) from the group.

    The height parameter specifies extraction height for this group:
    - float: explicit pixel value (or fraction if < 3)
    - "derived": compute from trace positions as span + fiber_diameter
    - None: use settings.json extraction_height (default)
    """

    range: tuple[int, int]  # [start, end) half-open interval, 1-based
    merge: str | list[int] = "center"  # "average", "center", or [indices]
    height: float | Literal["derived"] | None = None

    model_config = ConfigDict(extra="forbid")


class FiberBundleConfig(BaseModel):
    """Configuration for repeating fiber bundles.

    For instruments where fibers are organized in repeating bundles (e.g., 7 fibers
    per IFU target), this defines the bundle structure.

    Bundle assignment can use either:
    - Fixed size division (n_traces must be divisible by size)
    - bundle_centers: assign each trace to nearest bundle center (handles missing fibers)
    """

    size: int  # expected fibers per bundle
    count: int | None = None  # number of bundles (validated if provided)
    merge: str | list[int] = "center"  # "average", "center", or [indices]
    height: str | float | None = "derived"  # "derived", explicit pixels, or None

    # Bundle centers for robust assignment (handles missing fibers)
    bundle_centers: dict[int, float] | None = None  # inline: bundle_id -> y_position
    bundle_centers_file: str | list[str] | None = (
        None  # external file, per-channel list ok
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("bundle_centers", mode="before")
    @classmethod
    def convert_bundle_centers_keys(cls, v):
        """Convert string keys to int (YAML loads int keys as strings)."""
        if v is None:
            return v
        return {int(k): float(val) for k, val in v.items()}


# Type for trace selection in the 'use' section
# - "all": use all traces together
# - "groups": use traces with non-default group assignment
# - "per_fiber": separate processing per fiber_idx (for per-fiber wavecal)
# - list[str]: use specific groups by name
TraceSelection = Literal["all", "groups", "per_fiber"] | list[str]


class FibersConfig(BaseModel):
    """Configuration for multi-fiber instruments.

    Defines how physical fiber traces are organized into groups and which
    traces to use for each reduction step.

    For multi-order instruments (echelle), set fibers_per_order and provide
    order_centers to apply grouping within each spectral order.

    For multi-channel instruments, per-order fields can be lists indexed
    by channel (same pattern as orientation, extension, etc.).
    """

    groups: dict[str, FiberGroupConfig] | None = None
    bundles: FiberBundleConfig | None = None
    use: dict[str, TraceSelection] | None = (
        None  # step_name -> selection; "default" key for fallback
    )

    # Trace files separately by header value, then merge traces
    # e.g., trace_by: "FIBMODE" groups files by FIBMODE header (even/odd/etc)
    trace_by: str | None = None

    # Per-order grouping for echelle instruments
    # Setting fibers_per_order implies per-order organization
    # These can be lists for multi-channel instruments (indexed by channel)
    fibers_per_order: int | list[int] | None = None
    order_centers: dict[int, float] | None = None  # inline (single channel only)
    order_centers_file: str | list[str] | None = None  # per-channel list supported

    model_config = ConfigDict(extra="forbid")

    @field_validator("order_centers", mode="before")
    @classmethod
    def convert_order_centers_keys(cls, v):
        """Convert string keys to int (YAML loads int keys as strings)."""
        if v is None:
            return v
        return {int(k): float(val) for k, val in v.items()}

    @property
    def per_order(self) -> bool:
        """Whether per-order grouping is enabled (derived from fibers_per_order)."""
        return self.fibers_per_order is not None

    @model_validator(mode="after")
    def validate_per_order_config(self):
        """Ensure order_centers or order_centers_file when fibers_per_order is set."""
        if self.fibers_per_order is not None:
            if self.order_centers is None and self.order_centers_file is None:
                raise ValueError(
                    "order_centers or order_centers_file required when fibers_per_order is set"
                )
        return self


class InstrumentConfig(BaseModel):
    """Configuration for an astronomical instrument.

    This model validates the flat YAML structure used by instrument configs.
    It allows extra fields for instrument-specific parameters.
    """

    # Required identification
    __instrument__: str | None = None  # Internal name (uses alias due to dunder)
    instrument: str  # Header keyword for instrument name
    id_instrument: str  # Value/pattern to match

    # Telescope
    telescope: str | None = None

    # Date handling
    date: str = "DATE-OBS"
    date_format: str = "fits"

    # Channel system (detectors/optical paths)
    channels: list[str] | None = None
    channels_id: list[str] | None = None
    kw_channel: str | None = None
    id_channel: list[str] | None = None
    extension: int | str | list[int | str] = 0
    orientation: int | list[int] = 0
    transpose: bool = False

    # Multi-amplifier readout (optional)
    amplifiers: AmplifiersConfig | None = None

    # Multi-fiber configuration (optional)
    fibers: FibersConfig | None = None

    # Detector dimensions
    naxis_x: str | int = "NAXIS1"
    naxis_y: str | int = "NAXIS2"

    # Overscan/prescan regions
    prescan_x: int | str = 0
    overscan_x: int | str = 0
    prescan_y: int | str = 0
    overscan_y: int | str = 0

    # Calibration values (can be literals or header keywords)
    gain: float | int | str = 1
    readnoise: float | int | str = 0
    dark: float | int | str = 0
    sky: float | int | str = 0
    exposure_time: str = "EXPTIME"

    # Location (for barycentric correction)
    ra: str | None = "RA"
    dec: str | None = "DEC"
    longitude: float | str | None = None
    latitude: float | str | None = None
    altitude: float | str | None = None

    # Target identification
    target: str = "OBJECT"
    observation_type: str | None = None
    category: str | None = None
    image_type: str | None = None
    instrument_mode: str | None = None

    # File classification - header keywords
    kw_bias: str | None = None
    kw_flat: str | None = None
    kw_curvature: str | None = None
    kw_scatter: str | None = None
    kw_orders: str | None = None
    kw_wave: str | None = None
    kw_comb: str | None = None
    kw_spec: str | None = None

    # File classification - identifier patterns
    id_bias: str | None = None
    id_flat: str | None = None
    id_curvature: str | None = None
    id_scatter: str | None = None
    id_orders: str | None = None
    id_wave: str | None = None
    id_comb: str | None = None
    id_spec: str | None = None

    # Wavelength information
    wavelength_range: list | None = None
    wavecal_specifier: str | None = None

    # Allow additional fields for instrument-specific parameters
    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )

    @field_validator("channels", "channels_id", "id_channel", mode="before")
    @classmethod
    def ensure_list(cls, v):
        """Convert single values to lists."""
        if v is None:
            return None
        if isinstance(v, str):
            return [v]
        return v

    @field_validator("extension", "orientation", mode="before")
    @classmethod
    def normalize_list_or_scalar(cls, v):
        """Keep as-is, validation handles both forms."""
        return v


# Future models for the nested structure (REDESIGN.md)
# These will be used when migrating to the new architecture


class AmplifierConfig(BaseModel):
    """Configuration for a detector readout amplifier."""

    id: str
    gain: float | dict[str, str]  # literal or {key: "HEADER_KEY"}
    readnoise: float | dict[str, str]
    region: dict[str, list[int]] | None = None
    linearity: dict | None = None
    bad_pixel_mask: str | None = None

    model_config = ConfigDict(extra="forbid")


class DetectorConfig(BaseModel):
    """Configuration for a physical detector."""

    name: str
    naxis: tuple[int, int] | list[int]
    orientation: int = 0
    prescan: dict[str, list[int] | None] | None = None
    overscan: dict[str, list[int] | None] | None = None
    amplifiers: list[AmplifierConfig] = []
    bad_pixel_mask: str | None = None

    model_config = ConfigDict(extra="forbid")


class BeamArmConfig(BaseModel):
    """Configuration for a beam-splitter arm."""

    name: str
    polarization: str | None = None
    wavelength_shift: float = 0.0
    trace_offset: float = 0.0

    model_config = ConfigDict(extra="forbid")


class OpticalPathConfig(BaseModel):
    """Configuration for an optical path (fiber, slit position)."""

    name: str
    beam_arms: list[BeamArmConfig] | None = None

    model_config = ConfigDict(extra="forbid")


class DimensionConfig(BaseModel):
    """Configuration for a varying dimension (mode, fiber, etc.)."""

    values: list[str]
    header_key: str | None = None
    optional: bool = False

    model_config = ConfigDict(extra="forbid")


class InstrumentConfigV2(BaseModel):
    """Future nested instrument configuration structure.

    This model represents the target architecture from REDESIGN.md.
    Not yet used - will be activated when migrating instruments.
    """

    instrument: str
    telescope: str | None = None
    id_instrument: str

    detectors: list[DetectorConfig] = []
    optical_paths: list[OpticalPathConfig] = []
    dimensions: dict[str, DimensionConfig] = {}

    headers: dict[str, str] = {}
    file_types: dict[str, dict[str, str]] = {}

    model_config = ConfigDict(extra="allow")


def validate_instrument_config(data: dict[str, Any]) -> InstrumentConfig:
    """Validate instrument configuration data.

    Parameters
    ----------
    data : dict
        Raw configuration data (from YAML or JSON)

    Returns
    -------
    InstrumentConfig
        Validated configuration

    Raises
    ------
    pydantic.ValidationError
        If validation fails
    """
    return InstrumentConfig(**data)
