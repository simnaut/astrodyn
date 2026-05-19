# Common pitfalls

Subtle gotchas that produce wrong-but-plausible results if missed. The most
critical four live inline in `CLAUDE.md`; this page is the full catalog.

## Frame and reference-state gotchas

- **JEOD's `RefFrameState` stores position/velocity relative to parent frame,
  not global.** Don't confuse with absolute/inertial coordinates.
- **Gravity acceleration in `GravityInteraction` excludes the acceleration of
  the integration frame itself toward the planet.** For Earth-centered
  inertial integration, the Sun's contribution is the differential
  acceleration (vehicle toward Sun minus Earth toward Sun), not the absolute
  acceleration toward the Sun.
- **JEOD's `DynBody` has three reference frames**: `structure` (geometric
  origin), `composite_body` (composite CoM), `core_body` (this body's CoM
  only). State is integrated in one of these, then propagated to the others.

## Quaternion convention

- **JEOD uses left-transformation quaternions** (`r' = q r q*`). Many
  references use right-transformation. Getting this wrong produces the
  transpose rotation.

## Mass properties

- **`MassProperties.inertia` is about the body frame axes through the center
  of mass.** When composing masses, use the parallel axis theorem (Steiner's
  theorem) for the offset contribution.

## Trick / JEOD operational gotchas

- **Trick sim working directory**: JEOD sims must be run from the SIM root
  directory (e.g., `verif/SIM_dyncomp/`), not from `SET_test/RUN_*/`. The
  `input.py` files use paths like `SET_test/common_input.py` and
  `Log_data/log_suite.py` relative to the SIM root. Running from the wrong
  directory produces no data output.
- **Trick DRAscii silently drops unregistered variables**: when injecting
  ASCII logging snippets in `generate_references.sh`, variable names must
  match the S_define's object names exactly. If a variable doesn't exist in
  the sim, Trick silently omits it from the CSV — producing fewer columns
  than expected with no error message. Always verify the S_define (e.g.,
  SIM_2A_SHADOW_CALC uses `radiation_simple`, not `radiation`).

## Numerical instabilities at boundaries

- **Geodetic longitude at the poles**: at latitude ±90°, longitude is
  geometrically undefined (all meridians converge). `atan2(y, x)` becomes
  hypersensitive to position errors: at 89.8° latitude, ~3.7e-6 rad/m
  sensitivity. Polar orbit NED tests have larger longitude tolerances
  (~3.3e-5 rad) than inclined orbit tests (~6.5e-8 rad). This is not a code
  bug — both JEOD and our code produce valid but numerically unstable values.
