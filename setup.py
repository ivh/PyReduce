"""
Setup Module
Compiles the C functions
"""
from setuptools import setup


# from pyreduce.clib import build_cluster
# build_cluster.build()

from pyreduce.clib import build_extract

build_extract.build()

setup(name="pyreduce", version="0.0", packages=["pyreduce"])
