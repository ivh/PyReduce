"""
Setup Module
Compiles the C functions
"""
from setuptools import setup, find_packages

from pyreduce.clib import build_cluster, build_extract

build_cluster.build()
build_extract.build()

setup(name="pyreduce", version="0.0", packages=find_packages())
