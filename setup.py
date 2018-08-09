"""
Setup Module
Compiles the C functions
"""

import clib.build_cluster
import clib.build_extract


clib.build_cluster.build()
clib.build_extract.build()
