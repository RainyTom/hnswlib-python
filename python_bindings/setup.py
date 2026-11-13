"""Compatibility shim mirroring ``python_bindings/setup.py``.

The pure-Python package is configured by the repository ``pyproject.toml``;
this file exists so the binding directory keeps the upstream file layout.
"""

from setuptools import setup

setup()
