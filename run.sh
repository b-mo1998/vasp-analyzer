#!/bin/bash
# Launch VASP Analyzer.
cd "$(dirname "$0")" && exec .venv/bin/python -m vasp_analyzer "$@"
