# Architecture

The astrodyn workspace is structured around a strict three-layer dependency
rule. `CLAUDE.md` carries the one-paragraph summary; this page carries the
rationale, the exceptions, and the enforcement.

## Three-layer rule

All physics lives in **`astrodyn_*`** crates (pure Rust, zero Bevy dependency).
Orchestration lives in **`astrodyn`** — the workspace **root crate** (`src/` at
workspace root) — which composes `astrodyn_*` functions into pipeline stages
and re-exports all types; zero Bevy dependency. Bevy wiring lives in
**`astrodyn_bevy`** (`crates/astrodyn_bevy/` — thin glue: component derives,
systems that delegate to `astrodyn` functions, plugin registration).

`astrodyn_bevy` depends **only** on `astrodyn` + `bevy` — never on `astrodyn_*`
crates directly. **`astrodyn` is the single API surface for the production
path:** every Bevy system, every mission crate, every downstream consumer that
ships in a real simulation reads the workspace through `astrodyn` and only
through `astrodyn`. The narrower the production-path surface, the smaller the
contract that has to stay stable across phases.

This rule is scoped to the production path because the workspace also ships
a standalone arena-state simulation harness (`astrodyn_runner`) whose role
is the inverse: it owns its own state container and *needs* to construct
concrete physics types itself. See the next section for why that asymmetry is
intentional and what each consumer is allowed to depend on.

Never put physics algorithms directly in a Bevy system function. The system
queries components, then calls a `astrodyn` function. This keeps physics
portable to other ECS frameworks, WASM, or standalone batch computation.

## `astrodyn_bevy` vs `astrodyn_runner`: two parallel consumers of `astrodyn`

The titular simulation environment is **`astrodyn_bevy`** (`crates/astrodyn_bevy/`).
It is the production target — Bevy ECS is the chosen runtime for mission code,
and the ECS world is the single source of truth for all state.

**`astrodyn_runner` is a parallel non-Bevy consumer of `astrodyn`**, not a
dependency of `astrodyn_bevy`. It exists because the `astrodyn_*` and `astrodyn`
crates have **zero Bevy dependency** by design (the layer rule above), so
they can be exercised directly from a plain Rust binary that owns its own
state. `astrodyn_runner` provides that owned-state harness for:

- **Tier 3 cross-validation tests** (`crates/astrodyn_verif_jeod/tests/tier3_*.rs`)
  — propagating from JEOD initial conditions and comparing against
  Trick reference CSVs without standing up a Bevy `App`. The verif crate
  drives `astrodyn_runner::Simulation` end-to-end.
- **Batch propagation, scripting, and offline studies** that don't need
  ECS scheduling, parallelism, or Bevy plugins.

`astrodyn_runner` and `astrodyn_bevy` sit *next to* each other in the dep graph
— both depend on `astrodyn` and *only* `astrodyn` for physics; neither depends
on the other (except that `astrodyn_bevy` carries `astrodyn_runner` as a
dev-dep for parity-style tests). Any improvement that lands in `astrodyn_*` or
`astrodyn` (typed quantities, phantom-frame discipline, witness-gated
constructors like `BodyAttitude<V>`, …) benefits both consumers identically.

**Every workspace consumer of the `astrodyn` pipeline — mission crates,
`astrodyn_bevy`, `astrodyn_runner`, and the verification crates
(`astrodyn_verif_jeod`, `astrodyn_verif_parity`) — depends on
`astrodyn` and only `astrodyn` for physics** (+ `bevy` for the Bevy
adapter, + `astrodyn_runner` as a dev-dep on `astrodyn_bevy` for
parity tests, + `astrodyn_runner` and `astrodyn_verif_jeod` as
gateway-consumer deps for `astrodyn_verif_parity`). The "single API
surface" rule applies uniformly: every physics type, function, or
module that any consumer reaches must be reachable through `astrodyn`.
If something isn't, the fix is to widen `astrodyn`'s curated
re-export surface, not to add a direct `astrodyn_*` physics dep.

Owner-crate unit / Tier 2 / Tier 3 tests live inside their owning
crate's own `tests/` directory and reach the crate under test through
normal in-crate test access (no gateway, no verif crate). The verif
crates host the *cross-validation* against JEOD trajectories; they no
longer host kernel-level subsystem tests.

## Mission-crate consumption

Mission crates that target the production runtime depend on
`astrodyn_bevy` (and transitively `astrodyn`). They never depend on
`astrodyn_runner`, and they never reach around `astrodyn` to pull a
physics crate directly — if you find yourself writing
`astrodyn_dynamics = …` in a mission `Cargo.toml`, that is the bug.

## Enforcement

A CI lint (`scripts/check_no_bypass_deps.sh`) enforces the rule
structurally by failing the build if `astrodyn_runner`,
`astrodyn_bevy`, `astrodyn_verif_jeod`, or `astrodyn_verif_parity`
declare any direct `astrodyn_*` physics-crate dep.

## See also

- [Strategy](Strategy) — architecture and phase history (the big picture this rule
  serves).
- [Type-System](Type-System) — the typed quantities / phantom tags that travel
  through the layer boundaries.
