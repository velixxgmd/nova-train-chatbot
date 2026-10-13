#!/usr/bin/env python
"""Launch the Nova chat server (API + static UI)."""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
SRC = os.path.join(PROJECT, "src")
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "nova", "api"))

from nova.api.server import main  # noqa: E402


if __name__ == "__main__":
    main()
