"""Shared pytest configuration for BDH Graph Harness tests."""
import sys
import os

# Add the harness directory to sys.path so we can import harness.py
HARNESS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HARNESS_DIR not in sys.path:
    sys.path.insert(0, HARNESS_DIR)

import harness  # noqa: E402