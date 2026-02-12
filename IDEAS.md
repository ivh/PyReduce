# Ideas

- **Make extraction_height <1 be relative to Trace.height**: Replace `fix_extraction_height` neighbor-distance computation with a simple `fraction * trace.height` multiply, since trace.height already captures the per-trace order separation.
- **SlitFu from all orders**: For very low SNR, it would be nice to derive a fixed slit illumination from all spectral orders, by horizontal sum/median (taking trace+curve into acct). Then use that slitfu to extract.
- **Pyodide**: Run fully in users broser. CFFI supposedly works with this.
