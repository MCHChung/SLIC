"""Repository-relative locations used by the plotting scripts in this folder."""
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "data" / "results"            # analysis results, one folder per figure
EXP = REPO / "data" / "exp_raw"                # experimental sloshing-tank data
PLOTS = REPO / "plots"                         # figures are written here
PLOTS.mkdir(parents=True, exist_ok=True)
