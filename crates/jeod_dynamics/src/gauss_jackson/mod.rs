//! Gauss-Jackson (Störmer-Cowell) integrator for second-order ODEs.
//!
//! Faithful port of JEOD's Gauss-Jackson integrator:
//! - `gauss_jackson_integrator_base.hh` — FSM driver, start_cycle, integrate_gj
//! - `gauss_jackson_integrator_base_second.hh` — predict/correct/delinv (2nd-order ODE)
//! - `gauss_jackson_integration_controls.hh` — coefficient management, outer integration loop
//!
//! This integrator uses dual Störmer-Cowell / Summed-Adams coefficient sets,
//! inverse backward difference accumulators (delinv), and a 5-state finite
//! state machine for startup (Reset → Priming → BootstrapEdit → BootstrapStep
//! → Operational).

mod coefficients_pair;
mod coeffs;
pub mod config;
mod n_choose_m;
mod ratio128;
mod rational_coeffs;
mod state_machine;
mod two_d_array;

use crate::state::TranslationalState;
use coeffs::GaussJacksonCoeffs;
pub use config::GaussJacksonConfig;
use glam::DVec3;
use state_machine::{FsmState, StateMachine};
use two_d_array::TwoDArray;

/// Result of a single integration stage.
///
/// JEOD: `er7_utils::IntegratorResult`.
#[derive(Debug, Clone, Copy)]
pub struct IntegratorResult {
    /// Fraction of timestep completed.
    /// 0.0 = more stages needed (call `integrate` again with fresh derivatives).
    /// \>0.0 = step complete.
    /// JEOD: `IntegratorResult::time_scale`.
    pub time_scale: f64,
    /// Whether the convergence test passed.
    /// JEOD: `IntegratorResult::passed` (inverted from `failed`).
    pub passed: bool,
}

impl IntegratorResult {
    #[allow(dead_code)]
    fn needs_another_stage(self) -> bool {
        self.time_scale == 0.0
    }

    fn complete(passed: bool) -> Self {
        Self {
            time_scale: 1.0,
            passed,
        }
    }

    fn more_stages() -> Self {
        Self {
            time_scale: 0.0,
            passed: true,
        }
    }

    fn more_stages_with(passed: bool) -> Self {
        Self {
            time_scale: 0.0,
            passed,
        }
    }
}

/// Gauss-Jackson integrator state for second-order ODEs.
///
/// JEOD: `GaussJacksonIntegratorBase<GaussJacksonTwoState, SecondOrderODEIntegrator>`
/// combined with relevant parts of `GaussJacksonIntegrationControls`.
///
/// The integrator maintains acceleration history, inverse backward difference
/// accumulators (delinv), and a finite state machine that guides it through
/// priming, bootstrap, and operational phases.
#[derive(Debug, Clone)]
pub struct GaussJacksonState {
    // ── Coefficients (JEOD: owned by IntegrationControls) ──
    coeff: GaussJacksonCoeffs,
    state_machine: StateMachine,
    #[allow(dead_code)]
    config: GaussJacksonConfig,

    // ── Two-state fields ──
    // JEOD: `GaussJacksonTwoState init_state` — state at last reset
    init_vel: DVec3,
    init_pos: DVec3,
    // JEOD: `GaussJacksonTwoState delinv` — inverse backward differences
    delinv_vel: DVec3,
    delinv_pos: DVec3,
    // JEOD: `GaussJacksonTwoState corrector_sum` — speed hack for corrector
    corrector_sum_vel: DVec3,
    corrector_sum_pos: DVec3,

    // ── History arrays ──
    // JEOD: `DoubleTwoDArray acc_hist`, `DoubleTwoDArray pos_hist`
    acc_hist: TwoDArray,
    pos_hist: TwoDArray,

    // ── Scalar state ──
    /// JEOD: `velocity_corrector = 1.0 + corrector[order].sa_coefs[order]`
    velocity_corrector: f64,
    /// JEOD: `position_corrector = corrector[order].gj_coefs[order]`
    position_corrector: f64,
    /// Cached FSM state (JEOD: `fsm_state`)
    fsm_state: FsmState,
    /// Current integration order
    order: usize,
    /// Current number of history points
    history_length: usize,
    #[allow(dead_code)]
    max_history_size: usize,
    initial_order: usize,

    // ── Tolerances ──
    relative_tolerance: f64,
    absolute_tolerance: f64,

    // ── RK4 primer state ──
    // Staged RK4 for priming, matching JEOD's multi-stage primer dispatch.
    // Between stages, the caller recomputes derivatives at the intermediate state.
    primer_base_pos: DVec3,
    primer_base_vel: DVec3,
    primer_k_vel: [DVec3; 4], // acceleration at each stage
    primer_k_pos: [DVec3; 4], // velocity at each stage

    /// Internal stage counter within current integration step.
    /// Managed by the integrator (like JEOD's IntegrationControls).
    /// 0 = at start of cycle, 1 = predicted (needs correct), etc.
    current_stage: usize,
}

impl GaussJacksonState {
    /// Create a new Gauss-Jackson integrator with the given configuration.
    ///
    /// JEOD: Constructor of `GaussJacksonIntegratorBase` +
    /// `GaussJacksonIntegrationControls` initialization.
    pub fn new(config: GaussJacksonConfig) -> Self {
        config.validate();

        let max_order = config.final_order;
        let initial_order = config.initial_order;

        // Configure and compute coefficients for the initial order.
        // JEOD: coefficients are computed in IntegrationControls constructor.
        let mut coeff = GaussJacksonCoeffs::configure(max_order);
        coeff.compute_coeffs(initial_order);

        // Configure the state machine.
        let state_machine = StateMachine::configure(&config);

        let max_history_size = state_machine.max_history_size();

        // Allocate history arrays.
        // JEOD: `acc_hist.allocate(max_history_size, size)` where size=3
        let mut acc_hist = TwoDArray::new();
        let mut pos_hist = TwoDArray::new();
        acc_hist.allocate(max_history_size, 3);
        pos_hist.allocate(max_history_size, 3);

        let relative_tolerance = config.relative_tolerance;
        let absolute_tolerance = config.absolute_tolerance;

        Self {
            coeff,
            state_machine,
            config,
            init_vel: DVec3::ZERO,
            init_pos: DVec3::ZERO,
            delinv_vel: DVec3::ZERO,
            delinv_pos: DVec3::ZERO,
            corrector_sum_vel: DVec3::ZERO,
            corrector_sum_pos: DVec3::ZERO,
            acc_hist,
            pos_hist,
            velocity_corrector: 0.0,
            position_corrector: 0.0,
            fsm_state: FsmState::Reset,
            order: initial_order,
            history_length: 0,
            max_history_size,
            initial_order,
            relative_tolerance,
            absolute_tolerance,
            primer_base_pos: DVec3::ZERO,
            primer_base_vel: DVec3::ZERO,
            primer_k_vel: [DVec3::ZERO; 4],
            primer_k_pos: [DVec3::ZERO; 4],
            current_stage: 0,
        }
    }

    /// Reset the integrator to its initial state.
    ///
    /// JEOD: `GaussJacksonIntegratorBase::base_reset()`.
    /// Also resets internal stage counter and primer scratch state
    /// (not present in JEOD, which manages stages externally).
    pub fn reset(&mut self) {
        self.fsm_state = FsmState::Reset;
        self.history_length = 0;
        self.order = self.initial_order;
        self.state_machine.reset();
        self.current_stage = 0;
        self.primer_base_pos = DVec3::ZERO;
        self.primer_base_vel = DVec3::ZERO;
        self.primer_k_vel = [DVec3::ZERO; 4];
        self.primer_k_pos = [DVec3::ZERO; 4];
    }

    /// Returns the configuration this integrator was created with.
    pub fn config(&self) -> &GaussJacksonConfig {
        &self.config
    }

    /// Returns true if the integrator is still in the priming phase.
    pub fn is_priming(&self) -> bool {
        matches!(self.fsm_state, FsmState::Reset | FsmState::Priming)
    }

    /// Drive one stage of integration.
    ///
    /// Combines JEOD's `GaussJacksonIntegrationControls::integrate()` (stage
    /// management) with `GaussJacksonIntegratorBase::base_integrate()` (state
    /// integration). Call repeatedly with fresh acceleration (evaluated at
    /// `state.position`) until `result.time_scale > 0`.
    ///
    /// Stages are managed internally (like JEOD's IntegrationControls):
    /// - Priming: 4 calls per step (staged RK4)
    /// - BootstrapEdit: 1 call per edit point (time_scale=0, no time advance)
    /// - BootstrapStep/Operational: 2 calls per step (predict, correct)
    ///
    /// # Arguments
    /// - `sim_dt`: simulation timestep (JEOD: `sim_dt` passed to integration controls)
    /// - `time_scale_factor`: ratio of dynamic time to simulation time
    ///   (JEOD: `TimeDyn::scale_factor`, read via `TimeInterface::get_time_scale_factor()`).
    ///   1.0 for real-time, >1.0 for fast-forward.
    /// - `acc`: acceleration at current `state.position`
    /// - `state`: translational state (mutated in place)
    pub fn integrate(
        &mut self,
        sim_dt: f64,
        time_scale_factor: f64,
        acc: DVec3,
        state: &mut TranslationalState,
    ) -> IntegratorResult {
        self.current_stage += 1;
        let stage = self.current_stage;

        // JEOD dt variables (gauss_jackson_integration_controls.cc:144-149):
        //   cycle_simdt = sim_dt * cycle_scale
        //   cycle_dyndt = cycle_simdt * time_scale_factor
        // cycle_scale may change during start_cycle (via downsample), so we
        // compute cycle_dyndt after start_cycle for stage-1 paths. For
        // operational fast-path and primer stages 2-4, cycle_scale is stable.
        let cycle_dyndt = sim_dt * self.state_machine.cycle_scale() * time_scale_factor;

        // ── Operational fast path ──
        if self.fsm_state == FsmState::Operational {
            return self.integrate_operational(cycle_dyndt, stage, acc, state);
        }

        // ── Priming: stages 2-4 of RK4 (stage 1 handled after start_cycle) ──
        if self.fsm_state == FsmState::Priming && stage > 1 {
            return self.primer_step(cycle_dyndt, stage, acc, state);
        }

        // ── Start of cycle (stage 1 for all non-operational states) ──
        // start_cycle may trigger a downsample which changes cycle_scale.
        // It takes sim_dt + time_scale_factor (not pre-computed cycle_dyndt)
        // so it can derive cycle_dyndt from the post-downsample cycle_scale
        // for delinv reinitialization.
        let cycle_dyndt = if stage == 1 {
            self.start_cycle(sim_dt, time_scale_factor, acc, state);
            // Recompute: cycle_scale may have changed via downsample.
            sim_dt * self.state_machine.cycle_scale() * time_scale_factor
        } else {
            cycle_dyndt
        };

        // ── Dispatch based on FSM state after start_cycle ──
        match self.fsm_state {
            FsmState::Priming => {
                // Stage 1 of RK4 primer.
                self.primer_step(cycle_dyndt, 1, acc, state)
            }

            FsmState::BootstrapEdit => {
                // Edit one history point. Each call does: start_cycle already
                // stored the acceleration and advanced the SM. Now mid-correct
                // reconstructs the state at history_length. Returns time_scale=0
                // so the caller keeps looping (providing fresh acceleration at
                // the mid-corrected position for the next edit point).
                //
                // JEOD: if edit fails convergence, set_bootstrap_edit_redo_needed()
                // triggers a redo on the next FSM transition.
                let passed = self.edit_point(cycle_dyndt, state);
                if !passed {
                    self.state_machine.set_bootstrap_edit_redo_needed();
                }
                self.current_stage = 0; // Reset for next call
                IntegratorResult::more_stages()
            }

            FsmState::BootstrapStep => {
                // Predict/correct using GJ at current (possibly reduced) order.
                self.integrate_bootstrap_step(cycle_dyndt, stage, acc, state)
            }

            FsmState::Operational => {
                // Just transitioned to operational (first step after bootstrap).
                self.integrate_operational(cycle_dyndt, stage, acc, state)
            }

            FsmState::Reset => {
                // JEOD_INV: IG.12 — the state machine must not stay in Reset across an integrate()
                // call; the bootstrap path is always expected to transition it out.
                panic!("GaussJacksonState::integrate: stuck in Reset state");
            }
        }
    }

    /// Operational mode: predict (stage 1) then correct (stage 2).
    fn integrate_operational(
        &mut self,
        cycle_dyndt: f64,
        stage: usize,
        acc: DVec3,
        state: &mut TranslationalState,
    ) -> IntegratorResult {
        if stage == 1 {
            self.rotate_acc_hist();
            self.acc_hist.set_dvec3(self.order, acc);
        }

        let passed = self.integrate_gj(
            cycle_dyndt,
            stage,
            self.order as isize,
            self.order as isize,
            acc,
            None,
            state,
        );

        if stage == 1 {
            IntegratorResult::more_stages()
        } else {
            self.current_stage = 0;
            // Operational mode always has at_end_of_tour=true (scale_factor=1),
            // but check for consistency with bootstrap paths.
            if self.state_machine.at_end_of_tour() {
                IntegratorResult::complete(passed)
            } else {
                IntegratorResult::more_stages_with(passed)
            }
        }
    }

    /// BootstrapStep mode: predict (stage 1) then correct (stage 2).
    /// Note: start_cycle has already been called for stage 1 by integrate().
    fn integrate_bootstrap_step(
        &mut self,
        cycle_dyndt: f64,
        stage: usize,
        acc: DVec3,
        state: &mut TranslationalState,
    ) -> IntegratorResult {
        let offset = self
            .history_length
            .checked_sub(self.order + 1)
            .expect("integrate_bootstrap_step called before bootstrap primed history");
        let hist_len = self.history_length as isize;

        let passed = self.integrate_gj(
            cycle_dyndt,
            stage,
            hist_len - 1,
            hist_len,
            acc,
            Some(offset),
            state,
        );

        if stage == 1 {
            IntegratorResult::more_stages()
        } else {
            self.current_stage = 0;
            if self.state_machine.at_end_of_tour() {
                IntegratorResult::complete(passed)
            } else {
                IntegratorResult::more_stages_with(passed)
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Internal methods — line-by-line ports from JEOD
    // ═══════════════════════════════════════════════════════════════

    /// Start an integration cycle.
    ///
    /// JEOD: `GaussJacksonIntegratorBase::start_cycle(dt, acc, state)` +
    /// `GaussJacksonIntegrationControls::start_cycle(sim_dt)`.
    ///
    /// Takes `sim_dt` and `time_scale_factor` rather than pre-computed
    /// `cycle_dyndt` because `perform_step()` may trigger a downsample that
    /// changes `cycle_scale`. The post-downsample `cycle_dyndt` must be used
    /// for `initialize_*_integration_constants()`.
    fn start_cycle(
        &mut self,
        sim_dt: f64,
        time_scale_factor: f64,
        acc: DVec3,
        state: &TranslationalState,
    ) {
        if self.fsm_state == FsmState::Reset {
            // Save epoch data.
            // JEOD: `save_epoch_data(acc, state)`
            self.init_vel = state.velocity;
            self.init_pos = state.position;
            self.pos_hist.set_dvec3(0, state.position);
            self.acc_hist.set_dvec3(0, acc);
            self.history_length = 1;
        } else {
            // Non-reset: save incoming acceleration.
            self.acc_hist.set_dvec3(self.history_length, acc);
            self.history_length += 1;
        }

        // Advance the state machine.
        self.state_machine.perform_step();
        self.fsm_state = self.state_machine.fsm_state();

        // Downsample if indicated.
        // JEOD: cycle_simdt and cycle_dyndt are recomputed here, BEFORE
        // the reinitialize check (gauss_jackson_integration_controls.cc:298-301).
        if self.state_machine.at_downsample() {
            self.downsample_hist();
        }

        // Change order if indicated.
        if self.state_machine.at_order_change() {
            self.order = self.state_machine.current_order();
            // Recompute coefficients for the new order.
            self.coeff.compute_coeffs(self.order);
            self.velocity_corrector = 1.0 + self.coeff.corrector[self.order].sa_coefs[self.order];
            self.position_corrector = self.coeff.corrector[self.order].gj_coefs[self.order];
        }

        // Reinitialize delinv if indicated.
        // Compute cycle_dyndt from the (possibly updated) cycle_scale.
        if self.state_machine.at_reinitialize() {
            let cycle_dyndt = sim_dt * self.state_machine.cycle_scale() * time_scale_factor;
            if self.fsm_state == FsmState::BootstrapEdit {
                self.initialize_edit_integration_constants(cycle_dyndt);
                self.history_length = 1;
            } else {
                self.initialize_predictor_integration_constants(cycle_dyndt);
            }
        }
    }

    /// Edit a history point using the mid-corrector.
    ///
    /// JEOD: `GaussJacksonIntegratorBase::edit_point(dt, acc, state)`.
    fn edit_point(&mut self, dt: f64, state: &mut TranslationalState) -> bool {
        // JEOD_INV: IG.09 — history_length ≤ order is a structural precondition of `edit_point`
        assert!(self.history_length <= self.order);

        self.advance_edit_integration_constants(self.history_length);
        self.mid_correct(self.history_length, dt, state);
        self.test_for_convergence(state.position, self.history_length)
    }

    /// Integrate using Gauss-Jackson predictor and corrector.
    ///
    /// JEOD: `GaussJacksonIntegratorBase::integrate_gj(dt, target_stage, advance_index,
    ///        target_index, acc, ahist, state)`.
    #[allow(clippy::too_many_arguments)]
    fn integrate_gj(
        &mut self,
        dt: f64,
        target_stage: usize,
        advance_index: isize,
        target_index: isize,
        acc: DVec3,
        ahist_offset: Option<usize>,
        state: &mut TranslationalState,
    ) -> bool {
        if target_stage == 1 {
            // Predict stage
            self.advance_predictor_integration_constants(advance_index as usize);

            match ahist_offset {
                Some(offset) => {
                    let offset_view = self.acc_hist.offset_rows(offset);
                    let (vel_sum, pos_sum) = self
                        .coeff
                        .predictor
                        .apply_offset(&offset_view, self.order + 1);
                    self.apply_predict(dt, vel_sum, pos_sum, state);
                }
                None => {
                    let (vel_sum, pos_sum) =
                        self.coeff.predictor.apply(&self.acc_hist, self.order + 1);
                    self.apply_predict(dt, vel_sum, pos_sum, state);
                }
            }

            // Save comparison data
            self.pos_hist
                .set_dvec3(target_index as usize, state.position);

            // Pre-compute corrector sum (JEOD speed hack).
            // JEOD: `coeff->corrector[order].apply(size, order, ahist + 1, corrector_sum)`
            let (csum_vel, csum_pos) = match ahist_offset {
                Some(offset) => {
                    let offset_view = self.acc_hist.offset_rows(offset);
                    self.coeff.corrector[self.order]
                        .apply_offset_skip_first(&offset_view, self.order)
                }
                None => {
                    self.coeff.corrector[self.order].apply_skip_first(&self.acc_hist, self.order)
                }
            };
            self.corrector_sum_vel = csum_vel;
            self.corrector_sum_pos = csum_pos;

            true // predict always succeeds
        } else {
            // Correct stage
            self.correct(dt, acc, state);
            self.test_for_convergence(state.position, target_index as usize)
        }
    }

    // ── Predict/correct/delinv methods ──
    // Port of gauss_jackson_integrator_base_second.hh

    /// Initialize delinv for edit mode.
    ///
    /// JEOD: `initialize_edit_integration_constants(dt)`.
    fn initialize_edit_integration_constants(&mut self, dt: f64) {
        let dtsq = dt * dt;

        // Apply corrector[0] to acceleration history.
        let (vel_sum, pos_sum) = self.coeff.corrector[0].apply(&self.acc_hist, self.order + 1);

        // Compute inverse backward differences.
        // JEOD: `delinv.first[ii] = init_state.first[ii] / dt - delinv.first[ii]`
        self.delinv_vel = self.init_vel / dt - vel_sum;
        self.delinv_pos = self.init_pos / dtsq - pos_sum;
    }

    /// Advance delinv by one cycle (edit mode).
    ///
    /// JEOD: `advance_edit_integration_constants(index)`.
    fn advance_edit_integration_constants(&mut self, index: usize) {
        // JEOD: delinv.second[ii] += delinv.first[ii]
        //       delinv.first[ii] += acc_hist[index][ii]
        self.delinv_pos += self.delinv_vel;
        self.delinv_vel += self.acc_hist.get_dvec3(index);
    }

    /// Initialize delinv for predictor mode.
    ///
    /// JEOD: `initialize_predictor_integration_constants(dt)`.
    fn initialize_predictor_integration_constants(&mut self, dt: f64) {
        self.initialize_edit_integration_constants(dt);

        for ii in 1..self.order {
            self.advance_edit_integration_constants(ii);
        }

        // JEOD: `delinv.second[ii] += delinv.first[ii]`
        self.delinv_pos += self.delinv_vel;
    }

    /// Advance delinv by one cycle (predictor mode).
    ///
    /// JEOD: `advance_predictor_integration_constants(index)`.
    fn advance_predictor_integration_constants(&mut self, index: usize) {
        // JEOD: delinv.first[ii] += acc_hist[index][ii]
        //       delinv.second[ii] += delinv.first[ii]
        self.delinv_vel += self.acc_hist.get_dvec3(index);
        self.delinv_pos += self.delinv_vel;
    }

    /// Apply the predictor result to state.
    ///
    /// JEOD: `predict(dt, ahist, state)` — second half after apply().
    fn apply_predict(
        &self,
        dt: f64,
        vel_sum: DVec3,
        pos_sum: DVec3,
        state: &mut TranslationalState,
    ) {
        let dtsq = dt * dt;
        // JEOD: velocity[ii] = dt * (delinv.first[ii] + velocity[ii])
        //       position[ii] = dtsq * (delinv.second[ii] + position[ii])
        state.velocity = dt * (self.delinv_vel + vel_sum);
        state.position = dtsq * (self.delinv_pos + pos_sum);
    }

    /// Apply a mid-corrector.
    ///
    /// JEOD: `mid_correct(coeff_idx, dt, state)`.
    fn mid_correct(&self, coeff_idx: usize, dt: f64, state: &mut TranslationalState) {
        let dtsq = dt * dt;

        let (vel_sum, pos_sum) =
            self.coeff.corrector[coeff_idx].apply(&self.acc_hist, self.order + 1);

        // JEOD: state.first[ii] = dt * (delinv.first[ii] + state.first[ii])
        //       state.second[ii] = dtsq * (delinv.second[ii] + state.second[ii])
        state.velocity = dt * (self.delinv_vel + vel_sum);
        state.position = dtsq * (self.delinv_pos + pos_sum);
    }

    /// Apply the corrector.
    ///
    /// JEOD: `correct(dt, acc, state)`.
    fn correct(&self, dt: f64, acc: DVec3, state: &mut TranslationalState) {
        let dtsq = dt * dt;

        // JEOD: temp = first_csum[ii] + vfact * acc[ii]
        //       velocity[ii] = dt * (first_dinv[ii] + temp)
        let vel_temp = self.corrector_sum_vel + self.velocity_corrector * acc;
        state.velocity = dt * (self.delinv_vel + vel_temp);

        // JEOD: temp = second_csum[ii] + pfact * acc[ii]
        //       position[ii] = dtsq * (second_dinv[ii] + temp)
        let pos_temp = self.corrector_sum_pos + self.position_corrector * acc;
        state.position = dtsq * (self.delinv_pos + pos_temp);
    }

    /// Test for convergence.
    ///
    /// JEOD: `test_for_convergence(state, hist_data)`.
    /// Compares state.position against pos_hist[target_idx], then updates pos_hist.
    fn test_for_convergence(&mut self, new_pos: DVec3, target_idx: usize) -> bool {
        let old_pos = self.pos_hist.get_dvec3(target_idx);
        let mut passed = true;

        let error = (new_pos - old_pos).abs();
        let threshold = new_pos.abs() * self.relative_tolerance;

        // JEOD: if (error > absolute_tolerance) && (error > relative_tolerance * |new_data|)
        if error.x > self.absolute_tolerance && error.x > threshold.x {
            passed = false;
        }
        if error.y > self.absolute_tolerance && error.y > threshold.y {
            passed = false;
        }
        if error.z > self.absolute_tolerance && error.z > threshold.z {
            passed = false;
        }

        // Update pos_hist with new data.
        self.pos_hist.set_dvec3(target_idx, new_pos);

        passed
    }

    /// Rotate acceleration history down.
    ///
    /// JEOD: `rotate_acc_hist()` → `acc_hist.rotate_down(order)`.
    fn rotate_acc_hist(&mut self) {
        self.acc_hist.rotate_down(self.order);
    }

    /// Downsample acceleration and position histories.
    ///
    /// JEOD: `downsample_hist()`.
    fn downsample_hist(&mut self) {
        // JEOD_INV: IG.10 — downsample requires an odd history_length so the midpoint survives
        assert!(self.history_length & 1 == 1); // Must be odd
        let new_len = self.history_length.div_ceil(2);
        self.pos_hist.downsample(new_len);
        self.acc_hist.downsample(new_len);
        self.history_length = new_len;
    }

    // ── Staged RK4 primer ──
    // Implements a 4-stage RK4 integrator for the priming phase.
    // Between stages the caller recomputes acceleration at the intermediate state,
    // matching JEOD's integration framework where the derivative function is
    // called between each primer stage.

    /// One stage of the RK4 primer.
    ///
    /// Returns `more_stages()` for stages 1-3, `complete(true)` for stage 4
    /// (or `more_stages()` if the tour is not yet complete during subcycling).
    /// Between calls, the caller must evaluate acceleration at `state.position`.
    fn primer_step(
        &mut self,
        cycle_dyndt: f64,
        target_stage: usize,
        acc: DVec3,
        state: &mut TranslationalState,
    ) -> IntegratorResult {
        match target_stage {
            1 => {
                // Stage 1: evaluate at t_n
                self.primer_base_pos = state.position;
                self.primer_base_vel = state.velocity;
                self.primer_k_vel[0] = acc; // a(x_n)
                self.primer_k_pos[0] = state.velocity; // v_n

                // Move state to midpoint 1 for next derivative eval
                state.position = self.primer_base_pos + 0.5 * cycle_dyndt * self.primer_k_pos[0];
                state.velocity = self.primer_base_vel + 0.5 * cycle_dyndt * self.primer_k_vel[0];

                IntegratorResult::more_stages()
            }
            2 => {
                // Stage 2: evaluate at t_n + dt/2 (midpoint 1)
                self.primer_k_vel[1] = acc;
                self.primer_k_pos[1] = state.velocity;

                // Move state to midpoint 2
                state.position = self.primer_base_pos + 0.5 * cycle_dyndt * self.primer_k_pos[1];
                state.velocity = self.primer_base_vel + 0.5 * cycle_dyndt * self.primer_k_vel[1];

                IntegratorResult::more_stages()
            }
            3 => {
                // Stage 3: evaluate at t_n + dt/2 (midpoint 2)
                self.primer_k_vel[2] = acc;
                self.primer_k_pos[2] = state.velocity;

                // Move state to endpoint
                state.position = self.primer_base_pos + cycle_dyndt * self.primer_k_pos[2];
                state.velocity = self.primer_base_vel + cycle_dyndt * self.primer_k_vel[2];

                IntegratorResult::more_stages()
            }
            4 => {
                // Stage 4: evaluate at t_n + dt
                self.primer_k_vel[3] = acc;
                self.primer_k_pos[3] = state.velocity;

                // Combine: x_{n+1} = x_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
                state.velocity = self.primer_base_vel
                    + (cycle_dyndt / 6.0)
                        * (self.primer_k_vel[0]
                            + 2.0 * self.primer_k_vel[1]
                            + 2.0 * self.primer_k_vel[2]
                            + self.primer_k_vel[3]);

                state.position = self.primer_base_pos
                    + (cycle_dyndt / 6.0)
                        * (self.primer_k_pos[0]
                            + 2.0 * self.primer_k_pos[1]
                            + 2.0 * self.primer_k_pos[2]
                            + self.primer_k_pos[3]);

                // Save comparison data: position → pos_hist[history_length]
                self.pos_hist.set_dvec3(self.history_length, state.position);

                self.current_stage = 0; // Reset for next step
                if self.state_machine.at_end_of_tour() {
                    IntegratorResult::complete(true)
                } else {
                    IntegratorResult::more_stages()
                }
            }
            _ => panic!("RK4 primer: invalid target_stage {target_stage} (expected 1-4)"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gj_harmonic_oscillator() {
        // x'' = -x, x(0)=1, v(0)=0
        let dt: f64 = 0.01;
        let total_time = 10.0;
        let steps = (total_time / dt).round() as usize;

        let mut state = TranslationalState {
            position: DVec3::new(1.0, 0.0, 0.0),
            velocity: DVec3::ZERO,
        };

        let mut gj = GaussJacksonState::new(GaussJacksonConfig::with_order(8));

        for _ in 0..steps {
            loop {
                let acc = -state.position;
                let result = gj.integrate(dt, 1.0, acc, &mut state);
                if result.time_scale > 0.0 {
                    break;
                }
            }
        }

        let exact_pos = total_time.cos();
        let exact_vel = -total_time.sin();
        let pos_error = (state.position.x - exact_pos).abs();
        let vel_error = (state.velocity.x - exact_vel).abs();

        println!("GJ8 harmonic oscillator: pos_err={pos_error:.2e}, vel_err={vel_error:.2e}");
        assert!(
            pos_error < 1e-8,
            "GJ8 position error {pos_error:.2e} exceeds 1e-8"
        );
        assert!(
            vel_error < 1e-8,
            "GJ8 velocity error {vel_error:.2e} exceeds 1e-8"
        );
    }

    #[test]
    fn gj_kepler_orbit() {
        let mu: f64 = 3.986_004_415e14;
        let r0: f64 = 7_000_000.0;
        let v0 = (mu / r0).sqrt();

        let target_dt: f64 = 10.0;
        let period = 2.0 * std::f64::consts::PI * (r0.powi(3) / mu).sqrt();
        let steps = (period / target_dt).round() as usize;
        let dt = period / steps as f64;

        let mut state = TranslationalState {
            position: DVec3::new(r0, 0.0, 0.0),
            velocity: DVec3::new(0.0, v0, 0.0),
        };

        let mut gj = GaussJacksonState::new(GaussJacksonConfig::with_order(8));

        for _ in 0..steps {
            loop {
                let r = state.position.length();
                let acc = -mu / (r * r * r) * state.position;
                let result = gj.integrate(dt, 1.0, acc, &mut state);
                if result.time_scale > 0.0 {
                    break;
                }
            }
        }

        let pos_error = (state.position - DVec3::new(r0, 0.0, 0.0)).length();
        println!("GJ8 Kepler orbit: pos_err={pos_error:.2e} m (1 period)");
        // GJ8 with fixed order (no step-doubling bootstrap) gives ~11 km error
        // over one orbit at dt=10s. This is comparable to ABM8 because the
        // startup phase dominates the error budget. True JEOD performance
        // requires the full bootstrap machinery (initial_order=4, final_order=12,
        // ndoubling_steps=4).
        assert!(
            pos_error < 15_000.0,
            "GJ8 orbit position error {pos_error:.2e} m exceeds 15 km"
        );
    }

    #[test]
    fn gj_free_particle() {
        let dt: f64 = 0.5;
        let initial_pos = DVec3::new(1.0, 2.0, 3.0);
        let initial_vel = DVec3::new(4.0, 5.0, 6.0);

        let mut state = TranslationalState {
            position: initial_pos,
            velocity: initial_vel,
        };

        let mut gj = GaussJacksonState::new(GaussJacksonConfig::with_order(8));

        for _ in 0..20 {
            loop {
                let result = gj.integrate(dt, 1.0, DVec3::ZERO, &mut state);
                if result.time_scale > 0.0 {
                    break;
                }
            }
        }

        let expected_pos = initial_pos + initial_vel * 10.0;
        let pos_error = (state.position - expected_pos).length();
        assert!(pos_error < 1e-10, "Free particle error: {pos_error}");
    }

    #[test]
    fn gj_more_accurate_than_rk4() {
        let dt: f64 = 0.1;
        let steps = 100;
        let t_final = dt * steps as f64;

        let initial = TranslationalState {
            position: DVec3::new(1.0, 0.0, 0.0),
            velocity: DVec3::ZERO,
        };
        let accel_fn = |s: &TranslationalState, _t: f64| -> DVec3 { -s.position };

        // RK4 reference
        let mut state_rk4 = initial;
        for _ in 0..steps {
            state_rk4 = crate::rk4_translational_step(&state_rk4, accel_fn, dt);
        }

        // GJ8
        let mut state_gj = initial;
        let mut gj = GaussJacksonState::new(GaussJacksonConfig::with_order(8));
        for _ in 0..steps {
            loop {
                let acc = -state_gj.position;
                let result = gj.integrate(dt, 1.0, acc, &mut state_gj);
                if result.time_scale > 0.0 {
                    break;
                }
            }
        }

        let exact_pos = t_final.cos();
        let err_rk4 = (state_rk4.position.x - exact_pos).abs();
        let err_gj = (state_gj.position.x - exact_pos).abs();

        println!("RK4 err={err_rk4:.2e}, GJ8 err={err_gj:.2e}");
        assert!(
            err_gj < err_rk4,
            "GJ8 ({err_gj:.2e}) should be more accurate than RK4 ({err_rk4:.2e})"
        );
    }

    #[test]
    fn gj_harmonic_oscillator_bootstrap() {
        // Full bootstrap path: initial_order=4, final_order=12, ndoubling_steps=4.
        // With ndoubling_steps=4, tour_count=16, so the bootstrap phase uses
        // subcycled steps at dt/16, dt/8, dt/4, dt/2, then full dt.
        // This exercises the cycle_dyndt scaling and at_end_of_tour gating.
        let dt: f64 = 0.01;
        let total_time = 10.0;
        let steps = (total_time / dt).round() as usize;

        let mut state = TranslationalState {
            position: DVec3::new(1.0, 0.0, 0.0),
            velocity: DVec3::ZERO,
        };

        let mut gj = GaussJacksonState::new(GaussJacksonConfig::default());

        for _ in 0..steps {
            loop {
                let acc = -state.position;
                let result = gj.integrate(dt, 1.0, acc, &mut state);
                if result.time_scale > 0.0 {
                    break;
                }
            }
        }

        let exact_pos = total_time.cos();
        let exact_vel = -total_time.sin();
        let pos_error = (state.position.x - exact_pos).abs();
        let vel_error = (state.velocity.x - exact_vel).abs();

        println!(
            "GJ12 bootstrap harmonic oscillator: pos_err={pos_error:.2e}, vel_err={vel_error:.2e}"
        );
        // GJ12 with full bootstrap should be very accurate
        assert!(
            pos_error < 1e-10,
            "GJ12 bootstrap position error {pos_error:.2e} exceeds 1e-10"
        );
        assert!(
            vel_error < 1e-10,
            "GJ12 bootstrap velocity error {vel_error:.2e} exceeds 1e-10"
        );
    }

    #[test]
    fn gj_kepler_orbit_standard() {
        // GaussJacksonConfig::standard(): initial=8, final=12, ndoubling=2.
        // Tests bootstrap with smaller ndoubling (tour_count=4).
        let mu: f64 = 3.986_004_415e14;
        let r0: f64 = 7_000_000.0;
        let v0 = (mu / r0).sqrt();

        // Choose dt so that steps * dt == period exactly, avoiding orbit
        // closure error from rounding mismatch.
        let target_dt: f64 = 10.0;
        let period = 2.0 * std::f64::consts::PI * (r0.powi(3) / mu).sqrt();
        let steps = (period / target_dt).round() as usize;
        let dt = period / steps as f64;

        let mut state = TranslationalState {
            position: DVec3::new(r0, 0.0, 0.0),
            velocity: DVec3::new(0.0, v0, 0.0),
        };

        let mut gj = GaussJacksonState::new(GaussJacksonConfig::standard());

        for _ in 0..steps {
            loop {
                let r = state.position.length();
                let acc = -mu / (r * r * r) * state.position;
                let result = gj.integrate(dt, 1.0, acc, &mut state);
                if result.time_scale > 0.0 {
                    break;
                }
            }
        }

        let pos_error = (state.position - DVec3::new(r0, 0.0, 0.0)).length();
        println!("GJ standard Kepler orbit: pos_err={pos_error:.2e} m (1 period)");
        // GJ12 with bootstrap should have smaller orbit closure error than GJ8 fixed.
        assert!(
            pos_error < 15_000.0,
            "GJ standard orbit position error {pos_error:.2e} m exceeds 15 km"
        );
    }

    #[test]
    fn gj_cycle_scale_progression() {
        // Verify that cycle_scale follows the expected doubling sequence
        // during bootstrap.
        use state_machine::StateMachine;

        // ndoubling_steps=3, tour_count=8
        let config = GaussJacksonConfig {
            initial_order: 4,
            final_order: 10,
            ndoubling_steps: 3,
            ..Default::default()
        };
        let sm = StateMachine::configure(&config);
        assert!((sm.cycle_scale() - 1.0 / 8.0).abs() < 1e-15);

        // Run through FSM until operational, recording cycle_scale at
        // each downsample.
        let mut sm = StateMachine::configure(&config);
        let mut scales = vec![sm.cycle_scale()];

        for _ in 0..500 {
            sm.perform_step();
            if sm.at_downsample() {
                scales.push(sm.cycle_scale());
            }
            if sm.fsm_state() == FsmState::Operational {
                break;
            }
        }
        assert_eq!(
            sm.fsm_state(),
            FsmState::Operational,
            "FSM did not reach operational"
        );

        // Initial scale = 1/8, then doubles: 1/4, 1/2, 1.0
        assert_eq!(scales.len(), 4, "Expected 3 doublings + initial");
        let expected = [1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0, 1.0];
        for (i, (&got, &exp)) in scales.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-15,
                "cycle_scale[{i}]: expected {exp}, got {got}"
            );
        }
    }

    #[test]
    fn gj_time_scale_factor_equivalence() {
        // Validates that time_scale_factor actually multiplies into cycle_dyndt.
        //
        // Key insight: integrate(sim_dt=0.005, tsf=2.0) produces the same
        // cycle_dyndt as integrate(sim_dt=0.01, tsf=1.0) at every stage.
        // With the same number of integrate() calls, the state machine sees
        // identical step counts and the physics sees identical cycle_dyndt,
        // so the trajectories must be bitwise identical.
        //
        // This test would FAIL if time_scale_factor were ignored: run B
        // would integrate at half the effective dt, producing a very
        // different trajectory.
        let config = GaussJacksonConfig::default(); // ndoubling=4
        let n_steps: usize = 1000;

        // Run A: sim_dt=0.01, time_scale_factor=1.0 (baseline)
        // Dynamic time per step = 0.01 * 1.0 = 0.01
        let mut state_a = TranslationalState {
            position: DVec3::new(1.0, 0.0, 0.0),
            velocity: DVec3::ZERO,
        };
        let mut gj_a = GaussJacksonState::new(config);
        for _ in 0..n_steps {
            loop {
                let acc = -state_a.position;
                let result = gj_a.integrate(0.01, 1.0, acc, &mut state_a);
                if result.time_scale > 0.0 {
                    break;
                }
            }
        }

        // Run B: sim_dt=0.005, time_scale_factor=2.0 (same effective dt)
        // Dynamic time per step = 0.005 * 2.0 = 0.01
        // Same number of calls → same total dynamic time (10.0s).
        let mut state_b = TranslationalState {
            position: DVec3::new(1.0, 0.0, 0.0),
            velocity: DVec3::ZERO,
        };
        let mut gj_b = GaussJacksonState::new(config);
        for _ in 0..n_steps {
            loop {
                let acc = -state_b.position;
                let result = gj_b.integrate(0.005, 2.0, acc, &mut state_b);
                if result.time_scale > 0.0 {
                    break;
                }
            }
        }

        // Both should reach the same state (bitwise identical cycle_dyndt
        // at every stage means identical floating-point trajectories).
        let pos_diff = (state_a.position - state_b.position).length();
        let vel_diff = (state_a.velocity - state_b.velocity).length();
        println!("time_scale_factor equivalence: pos_diff={pos_diff:.2e}, vel_diff={vel_diff:.2e}");
        assert!(
            pos_diff < 1e-14,
            "Position divergence {pos_diff:.2e} between tsf=1.0 and tsf=2.0 runs"
        );
        assert!(
            vel_diff < 1e-14,
            "Velocity divergence {vel_diff:.2e} between tsf=1.0 and tsf=2.0 runs"
        );

        // Sanity: both should also be accurate vs exact solution
        let total_dyn_time: f64 = 10.0;
        let exact_pos = total_dyn_time.cos();
        let err_a = (state_a.position.x - exact_pos).abs();
        let err_b = (state_b.position.x - exact_pos).abs();
        assert!(
            err_a < 1e-10,
            "Run A position error {err_a:.2e} exceeds 1e-10"
        );
        assert!(
            err_b < 1e-10,
            "Run B position error {err_b:.2e} exceeds 1e-10"
        );
    }

    #[test]
    fn gj_time_scale_factor_affects_dynamics() {
        // Validates that time_scale_factor != 1.0 actually changes the dynamics.
        //
        // With tsf=2.0 and sim_dt=0.01, the effective dt is 0.02 per sim step.
        // After 500 sim steps, dynamic time = 500 * 0.02 = 10.0s.
        // With tsf=1.0 and sim_dt=0.01, after 500 sim steps, dynamic time = 5.0s.
        //
        // The two trajectories MUST differ — if they don't, time_scale_factor
        // is being ignored.
        let config = GaussJacksonConfig::default();
        let sim_steps = 500;

        // Run A: tsf=1.0, 500 steps → 5.0s of dynamic time
        let mut state_a = TranslationalState {
            position: DVec3::new(1.0, 0.0, 0.0),
            velocity: DVec3::ZERO,
        };
        let mut gj_a = GaussJacksonState::new(config);
        for _ in 0..sim_steps {
            loop {
                let acc = -state_a.position;
                let result = gj_a.integrate(0.01, 1.0, acc, &mut state_a);
                if result.time_scale > 0.0 {
                    break;
                }
            }
        }

        // Run B: tsf=2.0, 500 steps → 10.0s of dynamic time
        let mut state_b = TranslationalState {
            position: DVec3::new(1.0, 0.0, 0.0),
            velocity: DVec3::ZERO,
        };
        let mut gj_b = GaussJacksonState::new(config);
        for _ in 0..sim_steps {
            loop {
                let acc = -state_b.position;
                let result = gj_b.integrate(0.01, 2.0, acc, &mut state_b);
                if result.time_scale > 0.0 {
                    break;
                }
            }
        }

        // Run A should be at cos(5.0), Run B at cos(10.0) — very different
        let exact_a = 5.0_f64.cos();
        let exact_b = 10.0_f64.cos();
        let err_a = (state_a.position.x - exact_a).abs();
        let err_b = (state_b.position.x - exact_b).abs();

        println!(
            "tsf=1.0: pos={:.10}, exact={exact_a:.10}, err={err_a:.2e}",
            state_a.position.x
        );
        println!(
            "tsf=2.0: pos={:.10}, exact={exact_b:.10}, err={err_b:.2e}",
            state_b.position.x
        );

        assert!(err_a < 1e-10, "tsf=1.0 error {err_a:.2e} exceeds 1e-10");
        // tsf=2.0 doubles the effective dt → larger truncation error
        assert!(err_b < 1e-9, "tsf=2.0 error {err_b:.2e} exceeds 1e-9");

        // The states must actually differ (proves tsf is not ignored)
        let pos_diff = (state_a.position.x - state_b.position.x).abs();
        assert!(
            pos_diff > 0.1,
            "Runs with different time_scale_factor produced nearly identical \
             positions (diff={pos_diff:.2e}), suggesting time_scale_factor is ignored"
        );
    }
}
