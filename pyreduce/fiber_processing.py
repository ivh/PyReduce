# -*- coding: utf-8 -*-
"""
Module for processing and generating fiber traces based on a primary trace
and a fiber layout configuration.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)

def generate_fiber_traces(primary_traces, primary_column_ranges, fiber_layout_config, detector_num_cols=None):
    """
    Generates individual fiber traces from primary traces and a fiber layout configuration.

    Parameters
    ----------
    primary_traces : list or np.ndarray
        A list or array of polynomial coefficients for the primary traces.
        Each element is expected to be a list or array of coefficients
        [c_n, c_{n-1}, ..., c_0] for use with numpy.polyval.
    primary_column_ranges : list or np.ndarray
        A list or array of tuples, where each tuple (start_col, end_col)
        defines the column range for the corresponding primary trace.
    fiber_layout_config : dict
        The parsed 'fiber_layout' object from the instrument's JSON configuration.
        Expected structure:
        {
            "physical_order_groups": [
                {
                    "fibers": [
                        {"id": "FIBER_A", "spatial_offset": -5.0, "spectral_offset": 0.1},
                        {"id": "FIBER_B", "spatial_offset": 5.0}
                    ]
                },
                // ... more groups
            ]
        }
    detector_num_cols : int, optional
        The total number of columns on the detector. If provided, spectral offsets
        will be clamped to stay within [0, detector_num_cols-1].

    Returns
    -------
    all_fiber_traces : list
        A new list of polynomial coefficients for all individual fibers.
    all_fiber_column_ranges : list
        A new list of corresponding column ranges for all individual fibers.
    fiber_trace_mapping : list
        A list of dictionaries, where each dictionary maps a generated trace
        to its origin and fiber ID. Example:
        [
            {
                'physical_order_group_index': 0, // Index of the group in fiber_layout_config
                'fiber_id': 'FIBER_A',
                'original_primary_trace_index': 0, // Index in primary_traces
                'generated_trace_index': 0 // Index in all_fiber_traces
            },
            // ... more mappings
        ]
    """
    all_fiber_traces = []
    all_fiber_column_ranges = []
    fiber_trace_mapping = []
    generated_trace_overall_idx = 0

    if not fiber_layout_config or 'physical_order_groups' not in fiber_layout_config:
        logger.warning("Fiber layout configuration is missing or malformed. Returning primary traces.")
        # If no valid config, return primary traces as is, with basic mapping
        for i, trace in enumerate(primary_traces):
            all_fiber_traces.append(trace)
            all_fiber_column_ranges.append(primary_column_ranges[i])
            fiber_trace_mapping.append({
                'physical_order_group_index': -1, # Indicates no group used
                'fiber_id': f'primary_trace_{i}',
                'original_primary_trace_index': i,
                'generated_trace_index': generated_trace_overall_idx
            })
            generated_trace_overall_idx += 1
        return all_fiber_traces, all_fiber_column_ranges, fiber_trace_mapping

    num_primary_traces = len(primary_traces)
    num_physical_order_groups = len(fiber_layout_config['physical_order_groups'])

    if num_primary_traces != num_physical_order_groups:
        logger.warning(
            f"Mismatch between number of primary traces ({num_primary_traces}) "
            f"and physical_order_groups in fiber_layout ({num_physical_order_groups}). "
            "Processing based on the shorter of the two."
        )
        # Potentially handle this more gracefully or raise an error based on stricter requirements later.
        # For now, iterate up to the minimum of the two.
        num_groups_to_process = min(num_primary_traces, num_physical_order_groups)
    else:
        num_groups_to_process = num_primary_traces

    for i in range(num_groups_to_process):
        primary_trace_coeffs = np.array(primary_traces[i])
        primary_col_start, primary_col_end = primary_column_ranges[i]
        
        group = fiber_layout_config['physical_order_groups'][i]
        
        if 'fibers' not in group or not group['fibers']:
            logger.warning(f"No fibers defined in physical_order_group index {i}. Skipping this group.")
            # Optionally, could add the primary trace itself if no fibers are defined for its group
            # all_fiber_traces.append(primary_trace_coeffs)
            # all_fiber_column_ranges.append((primary_col_start, primary_col_end))
            # fiber_trace_mapping.append({
            #     'physical_order_group_index': i,
            #     'fiber_id': f'primary_unassigned_{i}',
            #     'original_primary_trace_index': i,
            #     'generated_trace_index': generated_trace_overall_idx
            # })
            # generated_trace_overall_idx += 1
            continue

        for fiber_info in group['fibers']:
            fiber_id = fiber_info['id']
            spatial_offset = fiber_info.get('spatial_offset', 0.0)
            spectral_offset = fiber_info.get('spectral_offset', 0.0)

            # Clone and adjust polynomial for spatial offset
            # Coeffs are [c_n, ..., c_1, c_0], so c_0 is the last term
            new_trace_coeffs = primary_trace_coeffs.copy()
            new_trace_coeffs[-1] += spatial_offset
            all_fiber_traces.append(new_trace_coeffs.tolist())

            # Adjust column range for spectral offset
            new_col_start = int(round(primary_col_start + spectral_offset))
            new_col_end = int(round(primary_col_end + spectral_offset))

            if detector_num_cols is not None:
                new_col_start = max(0, new_col_start)
                new_col_end = min(detector_num_cols -1 , new_col_end)
                # Ensure start is not greater than end after clamping
                if new_col_start > new_col_end:
                    logger.warning(
                        f"Fiber {fiber_id} in group {i} has adjusted column range "
                        f"[{new_col_start}, {new_col_end}] which is invalid after clamping. "
                        f"Original primary range: [{primary_col_start}, {primary_col_end}], "
                        f"spectral offset: {spectral_offset}. Setting to empty range [0,0)."
                    )
                    new_col_start = 0
                    new_col_end = 0 # Results in an empty range

            all_fiber_column_ranges.append((new_col_start, new_col_end))

            # Store mapping information
            fiber_trace_mapping.append({
                'physical_order_group_index': i,
                'fiber_id': fiber_id,
                'original_primary_trace_index': i,
                'generated_trace_index': generated_trace_overall_idx
            })
            generated_trace_overall_idx += 1
            
    # If some primary traces were not processed due to fewer physical_order_groups
    if num_primary_traces > num_physical_order_groups:
        logger.info(f"Adding remaining {num_primary_traces - num_physical_order_groups} primary traces without fiber layout modifications.")
        for i in range(num_physical_order_groups, num_primary_traces):
            all_fiber_traces.append(primary_traces[i])
            all_fiber_column_ranges.append(primary_column_ranges[i])
            fiber_trace_mapping.append({
                'physical_order_group_index': -1, # Indicates no group used
                'fiber_id': f'primary_unmatched_{i}',
                'original_primary_trace_index': i,
                'generated_trace_index': generated_trace_overall_idx
            })
            generated_trace_overall_idx += 1

    return all_fiber_traces, all_fiber_column_ranges, fiber_trace_mapping
