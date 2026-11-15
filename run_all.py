"""Runs the full pipeline in order: python run_all.py"""
import runpy
import sys
from pathlib import Path

scripts = Path(__file__).resolve().parent / "scripts"
sys.path.insert(0, str(scripts))
for script in sorted(scripts.glob("0*.py")):
    print(f"\n{'=' * 70}\n{script.name}\n{'=' * 70}")
    runpy.run_path(str(script), run_name="__main__")
