# Conformal bounds for hydrogen damage in steels (pilot)

Pilot scripts testing whether conformal prediction bounds hold on held-out tests,
not just on average.

## Scripts
| File | What it does |
|---|---|
| `pilot.py` | Per-group coverage on HE-Austenite v0.1 (109 tests, austenitic stainless steels) |
| `fcg_pilot_v2.py` | Fatigue crack growth in gaseous H2 (X52, X70, X100), ASME B31.12 CC220 curve as baseline |
| `fcg_checks_v3.py` | Fair held-out-curve comparison (curve-level CV+) and code-curve exceedance with tolerance |
| `fcg_shift_v4.py` | Pressure-shift test: hold out each grade's highest-pressure curve |
| `x100_fig3_check.csv` | Points read from the NIST X100 figure, used to validate the digitized data |

## Main pilot result
On 8 hydrogen crack-growth curves, a one-sided 90% upper bound on da/dN covers about 90%
of points on average across held-out curves. On the held-out X100 curve at 20.68 MPa it
covers 36% (gradient boosting) and 70% (ASME curve). Average coverage hides failures on
individual tests. This is a pilot on 8 curves, not a guarantee.

## Data (not included)
- HE-Austenite v0.1: github.com/mfaizansworkspace-ai/Hydrogen-Embrittlement-in-Austenetic-SS (CC BY 4.0)
- Digitized crack-growth curves: github.com/tRosJan/machine-learning-hydrogen-assisted-fatigue (MIT)
- Primary sources: NIST (Amaro, Drexler, Slifka et al.) fatigue crack growth tests in gaseous hydrogen
- ASME curve equations: San Marchi et al., PVP2024-122529 (technical basis paper)

## Run
    pip install numpy pandas scikit-learn
    python pilot.py
    python fcg_pilot_v2.py "path/to/fcg/dataset"
    python fcg_checks_v3.py "path/to/fcg/dataset"
    python fcg_shift_v4.py "path/to/fcg/dataset"

## Known limits
- X70 at 5.5 MPa mixes two steels (X70A, X70B) in the digitized file; results on that curve are provisional.
- 8 curves are too few for a curve-level coverage guarantee.

Irene Boruah, ORCID 0009-0000-2094-8251
