# Fiber Layout Specification

This document outlines the JSON structure, integration, usage, and data products of the `fiber_layout` setting in PyReduce.

## JSON Structure

The `fiber_layout` setting is an optional object within the `instrument` block of a PyReduce instrument settings JSON file. Its primary purpose is to describe the physical arrangement of fibers relative to each other, specifically for instruments where multiple fibers are imaged onto the detector simultaneously, potentially within the same spectral orders.

The structure is defined as follows, nested within the `instrument` configuration:

```json
{
  "__instrument__": "MySpectrograph",
  // ... other top-level settings like "reduce", "bias", "flat" ...
  "instrument": {
    "gain": 1.0,
    "readnoise": 3.5,
    // ... other instrument-specific parameters ...
    "fiber_layout": {
      "physical_order_groups": [
        {
          "fibers": [
            {
              "id": "FIBER_A_ID",
              "spatial_offset": -5.2,
              "spectral_offset": 0.1
            },
            {
              "id": "FIBER_B_ID",
              "spatial_offset": 6.1
            }
          ]
        },
        {
          "fibers": [
            {
              "id": "FIBER_C_ID",
              "spatial_offset": 0.0 
            }
          ]
        }
      ]
    }
  }
  // ... other step settings ...
}
```

**Explanation of fields:**

*   **`instrument`**: The root object for instrument-specific configurations.
*   **`fiber_layout`**: The root object for the fiber layout configuration, nested within `instrument`.
*   **`physical_order_groups`**: An array of objects. Each object in this array represents a group of fibers whose traces are expected to be found close to each other on the detector. The grouping implies that these fibers likely share the same physical echelle order or are otherwise spatially related.
*   **`fibers`**: An array of fiber objects within each `physical_order_group`.
    *   **`id`** (string, required): A unique identifier for the fiber (e.g., "SKY", "SCIENCE_1", "CALIB_A"). This ID will be used to label the extracted spectra and in filenames.
    *   **`spatial_offset`** (number, required): The offset of this fiber in the spatial direction (cross-dispersion) relative to a reference point within its `physical_order_group`. The units are pixels. A positive value typically means an offset upwards (increasing row index) on the detector, and a negative value means downwards (decreasing row index).
    *   **`spectral_offset`** (number, optional): The offset of this fiber in the spectral direction (dispersion) relative to a reference point within its `physical_order_group`. The units are pixels. A positive value typically means an offset to the right (increasing column index) on the detector, and a negative value means to the left (decreasing column index). If not provided, it defaults to `0.0`.

## Integration into an Existing Instrument's JSON File

The `fiber_layout` object should be added as a key within the main `"instrument"` object in an instrument's specific settings JSON file (e.g., `settings_MYINSTRUMENT.json`).

**Example:**

```json
{
  "__instrument__": "MySpectrograph",
  "reduce": {
    "base_dir": "/data/my_instrument",
    "input_dir": "{instrument}/{night}",
    "output_dir": "{instrument}/{night}/reduced"
  },
  "instrument": { // Root instrument configuration block
    "gain": 1.0,
    "readnoise": 3.5,
    // ... other instrument specific settings like "date", "target", "kw_modes" etc. ...
    "fiber_layout": { // fiber_layout nested here
      "physical_order_groups": [
        { 
          // First group of traces expected 
          // (e.g., corresponds to the first primary trace found by trace_orders)
          "fibers": [
            {
              "id": "SCI", // Reference fiber for this group
              "spatial_offset": 0.0 
            },
            {
              "id": "SKY1",
              "spatial_offset": -25.5, // Sky fiber 25.5 pixels below SCI
              "spectral_offset": -0.5  // and slightly shifted left
            }
          ]
        },
        { 
          // Second group of traces expected
          "fibers": [
            {
              "id": "CAL", // Reference fiber for this group
              "spatial_offset": 0.0 
            }
          ]
        }
        // ... more groups if applicable
      ]
    }
  },
  "bias": {
    // ... bias settings ...
  }
  // ... other reduction step settings ...
}
```

## Workflow and Data Propagation

1.  **Configuration Loading:** The `fiber_layout` object is loaded as part of the instrument configuration by `pyreduce.configuration.load_config` and made available via the `Instrument.get_fiber_layout()` method.
2.  **Order Tracing (`OrderTracing` step):**
    *   The `OrderTracing.run` method in `pyreduce/reduce.py` calls `pyreduce.trace_orders.mark_orders` to find primary traces on the detector.
    *   It then retrieves the `fiber_layout` configuration.
    *   If a valid `fiber_layout` is found, it calls `pyreduce.fiber_processing.generate_fiber_traces`. This function takes the primary traces and the `fiber_layout` to generate new polynomial coefficients and column ranges for each individual fiber defined in the layout.
    *   The output of `generate_fiber_traces` includes:
        *   `all_fiber_traces`: A list of polynomial coefficients for all individual fibers.
        *   `all_fiber_column_ranges`: Corresponding column ranges for these fiber traces.
        *   `fiber_trace_mapping`: A list of dictionaries, detailing the origin of each generated trace (original primary trace index, physical order group index, fiber ID, and its new generated trace index).
    *   The `OrderTracing` step then saves `all_fiber_traces` (as `orders`), `all_fiber_column_ranges` (as `column_range`), and `fiber_trace_mapping` into its output `.npz` file (e.g., `PREFIX.ord_default.npz`).
3.  **Subsequent Steps:**
    *   All subsequent reduction steps that depend on the `'orders'` output (e.g., `BackgroundScatter`, `NormalizeFlatField`, `ScienceExtraction`) will receive the potentially expanded list of fiber traces and their column ranges, along with the mapping information.
    *   These steps will operate on each fiber trace individually. For example, `ScienceExtraction` will extract a spectrum for each defined fiber.

## Mapping `physical_order_groups` to Traces from `trace_orders.py`

The mapping between the `physical_order_groups` defined in the JSON and the primary traces identified by `trace_orders.py` is based on the **order of appearance**:

1.  `trace_orders.py` first identifies a set of primary traces on the detector. The exact number and position depend on the instrument, source brightness, and `orders` step settings.
2.  The `physical_order_groups` array in the `fiber_layout` JSON is expected to correspond **sequentially** to these primary traces.
    *   The first object in `physical_order_groups` corresponds to the first primary trace found.
    *   The second object corresponds to the second primary trace, and so on.
3.  Within each `physical_order_group`, one fiber **must** have `spatial_offset: 0.0` (and implicitly `spectral_offset: 0.0` if not specified). This fiber is the **reference fiber** for that group. The trace identified by `trace_orders.py` for this group is assumed to be the trace of this reference fiber.
4.  Other fibers within the same group are located by applying their `spatial_offset` and `spectral_offset` relative to this reference fiber's trace.
5.  The `generate_fiber_traces` function handles cases of mismatch between the number of found primary traces and defined `physical_order_groups` by processing the minimum of the two and logging warnings. Unmatched primary traces might be passed through without fiber processing, or unmatched groups ignored.

## Definition of Offsets

*   **`spatial_offset`**:
    *   **Units**: Pixels.
    *   **Direction**: Perpendicular to the dispersion direction (cross-dispersion).
        *   A **positive** value typically means an offset towards larger detector row indices ("up").
        *   A **negative** value typically means an offset towards smaller detector row indices ("down").
    *   **Reference**: Relative to the reference fiber's trace within the *same* `physical_order_group`. The offset is added to the constant term (c0) of the polynomial describing the reference trace.

*   **`spectral_offset`**:
    *   **Units**: Pixels.
    *   **Direction**: Along the dispersion direction.
        *   A **positive** value typically means an offset towards larger detector column indices ("right").
        *   A **negative** value typically means an offset towards smaller detector column indices ("left").
    *   **Reference**: Relative to the reference fiber's trace within the *same* `physical_order_group`. The offset is added to the start and end column of the `column_range` for that fiber.

## Output Files and FITS Headers

When a `fiber_layout` is used, the `Finalize` step in PyReduce aims to produce one `.ech` (echelle format) FITS file for each science fiber that was processed.

*   **Filename Convention:** The output filenames will typically include the fiber ID, for example: `original_input_SCI.ech`, `original_input_SKY1.ech`. The exact format is determined by the `finalize.filename` setting in the configuration, which is modified to include the fiber ID.
*   **FITS Header Keywords:** Each per-fiber `.ech` file will contain the following specific HIERARCH keywords in its primary HDU header to provide traceability:
    *   `HIERARCH PR FIBER_ID`: (String) The `id` of the fiber as defined in the `fiber_layout` (e.g., "SCI", "SKY1").
    *   `HIERARCH PR POG_IDX`: (Integer) The index of the `physical_order_group` (0-based) in the `fiber_layout.physical_order_groups` array to which this fiber belongs.
    *   `HIERARCH PR OGT_IDX`: (Integer) The index of the original primary trace (0-based) found by `trace_orders.py` that served as the reference for this fiber's physical order group.
    *   `HIERARCH PR GFT_IDX`: (Integer) The unique index (0-based) of this specific fiber's trace within the full list of all generated fiber traces produced by `generate_fiber_traces`. This corresponds to its row index in the `orders`, `column_range`, and `fiber_trace_mapping` arrays.

These keywords allow users to identify each fiber and relate it back to the `fiber_layout` configuration and the intermediate data products.

## Assumptions

1.  **Primary Trace Identification**: `trace_orders.py` successfully identifies the primary traces for the reference fiber of each `physical_order_group`.
2.  **Offset Consistency**: Pixel offsets are consistent along the spectral order. Significant variations (e.g., due to strong distortions making fibers non-parallel) are not handled by simple scalar offsets.
3.  **Sufficient Separation for Tracing**: `trace_orders.py` can distinguish primary traces for each group. The `fiber_layout` does not deblend traces merged by `trace_orders.py`.
4.  **Order of Traces**: The order of primary traces from `trace_orders.py` is stable and maps sequentially to `physical_order_groups`.
5.  **No Dynamic Remapping**: Mapping is static based on initial trace detection.
6.  **Offsets Apply to Fitted Polynomials**: Offsets are applied to the fitted polynomial of the reference trace.

This specification provides a framework for handling multi-fiber data where the relative positions of fibers are known and stable.
