# Ideas

- **Make extraction_height <1 be relative to Trace.height**: Replace `fix_extraction_height` neighbor-distance computation with a simple `fraction * trace.height` multiply, since trace.height already captures the per-trace order separation.
