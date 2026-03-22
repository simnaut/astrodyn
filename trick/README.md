# Trick Container for JEOD Reference Data

Builds NASA's [Trick](https://github.com/nasa/trick) simulation framework and
[JEOD](https://github.com/nasa/jeod) inside a Rocky Linux 9 container, then
runs JEOD verification sims to produce CSV reference trajectories for
bevy_jeod's Tier 3 cross-validation tests.

## Prerequisites

- Docker (or Podman)
- `../trick` — Trick source checkout
- `../jeod` — JEOD source checkout

## Build

From the `bevy_jeod` project root:

```bash
# Build the container (context is parent dir so trick/ and jeod/ are accessible)
docker build -f trick/Dockerfile -t jeod-trick ..
```

This takes 15-30 minutes (compiling Trick + JEOD from source).

## Generate Reference Data

```bash
# Run all JEOD verification sims and export CSVs to test_data/
mkdir -p test_data
docker run --rm -v $(pwd)/test_data:/output jeod-trick
```

The container runs these JEOD sims:

| Sim | Run | Duration | Validates |
|-----|-----|----------|-----------|
| SIM_dyncomp | RUN_2 | 28800s (8h) | Translational dynamics, spherical gravity |
| SIM_orbinit | RUN_0001 | instant | Orbital element initialization |
| SIM_Euler | RUN_inc | 86400s (24h) | Euler angle derived state |
| SIM_OrbElem | RUN_circular | 86400s | Orbital element computation |
| SIM_integ_test | RUN_rk4 | 28800s | RK4 integrator accuracy |

## Output

CSV files appear in `test_data/`:

```
test_data/
  dyncomp_run2_state.csv         # ISS 8-hour trajectory
  dyncomp_run2_Earth_RNP.csv     # Earth RNP data
  ...
```

The generate script canonicalizes JEOD output filenames by stripping the `log_` prefix
and `_ASCII` suffix (e.g., `log_state_ASCII.csv` becomes `state.csv`, prepended with
the sim label). The Tier 3 test expects `dyncomp_run2_state.csv`.

These are consumed by `cargo test` when the `test_data/` directory is present.

## Manual Usage

To get a shell inside the container for interactive sim runs:

```bash
docker run --rm -it -v $(pwd)/test_data:/output jeod-trick bash
cd /jeod/verif/SIM_dyncomp
trick-CP                                    # compile
./S_main*.exe SET_test/RUN_2/input.py        # run from SIM root
```

## Rocky 9 Package List

From Trick's CI (`test_linux.yml` Rocky 9 matrix entry):

```
epel-release, bison, clang, clang-devel, cmake, diffutils, flex,
gcc, gcc-c++, git, java-21-openjdk-devel, libxml2-devel, llvm,
llvm-devel, llvm-static, make, maven, ncurses-devel, openmotif,
openmotif-devel, perl, perl-Digest-MD5, python3-devel, swig,
udunits2, udunits2-devel, which, zlib-devel, zip, gdb,
gtest-devel, gmock-devel
```

CRB (CodeReady Builder) repo must be enabled for gtest/gmock.
