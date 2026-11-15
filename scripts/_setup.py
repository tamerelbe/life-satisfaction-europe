"""Makes `src` importable when a script is run from the repo root or scripts/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
