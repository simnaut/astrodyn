//! Gauss-Jackson finite state machine.
//!
//! Port of JEOD's `GaussJacksonStateMachine`
//! (`gauss_jackson_state_machine.hh/cc`).
//!
//! Guides the Gauss-Jackson integration through five phases:
//! Reset → Priming → BootstrapEdit → BootstrapStep → Operational.

use super::config::GaussJacksonConfig;

/// Finite state machine states for Gauss-Jackson integration.
///
/// JEOD: `GaussJacksonStateMachine::FsmState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FsmState {
    /// Module was just commanded to reset itself.
    Reset,
    /// Using primer to build initial set of data.
    Priming,
    /// Editing primer / lower-level Gauss-Jackson data.
    BootstrapEdit,
    /// Building toward downsample / change in order.
    BootstrapStep,
    /// At desired rate and order.
    Operational,
}

/// Gauss-Jackson finite state machine.
///
/// JEOD: `GaussJacksonStateMachine` in `gauss_jackson_state_machine.hh`.
#[derive(Debug, Clone)]
pub(crate) struct StateMachine {
    // Configuration (set once by configure())
    initial_order: usize,
    final_order: usize,
    #[allow(dead_code)]
    ndoubling_steps: usize,
    max_correction_iterations: usize,
    max_history_size: usize,
    tour_count: usize,

    // Runtime state
    fsm_state: FsmState,
    current_order: usize,
    history_size: usize,
    history_length: usize,
    scale_factor: usize,
    step_increment: usize,
    steps_since_reset: usize,
    correction_iterations: usize,
    cycle_scale: f64,
    cycle_start_time: f64,
    bootstrap_edit_redo_needed: bool,

    // Flags (set by perform_step, read by integrator)
    at_downsample: bool,
    at_reinitialize: bool,
    at_order_change: bool,
    at_end_of_tour: bool,
}

impl StateMachine {
    /// Configure the state machine from a GaussJacksonConfig.
    ///
    /// JEOD: `GaussJacksonStateMachine::configure(config)`.
    pub fn configure(config: &GaussJacksonConfig) -> Self {
        let initial_order = config.initial_order;
        let final_order = config.final_order;
        let ndoubling_steps = config.ndoubling_steps;
        let max_correction_iterations = config.max_correction_iterations;

        let last_doubling_order = if ndoubling_steps != 0 {
            final_order.min(initial_order + 2 * ndoubling_steps)
        } else {
            0
        };
        let max_history_size = (2 * last_doubling_order).max(final_order) + 1;
        let tour_count = 1usize << ndoubling_steps;

        let mut sm = Self {
            initial_order,
            final_order,
            ndoubling_steps,
            max_correction_iterations,
            max_history_size,
            tour_count,
            // Runtime state (set by reset)
            fsm_state: FsmState::Reset,
            current_order: initial_order,
            history_size: 1,
            history_length: 0,
            scale_factor: tour_count,
            step_increment: 1,
            steps_since_reset: 0,
            correction_iterations: 0,
            cycle_scale: 1.0 / tour_count as f64,
            cycle_start_time: 0.0,
            bootstrap_edit_redo_needed: false,
            at_downsample: false,
            at_reinitialize: false,
            at_order_change: false,
            at_end_of_tour: false,
        };
        sm.reset();
        sm
    }

    // ── Getters ──

    pub fn fsm_state(&self) -> FsmState {
        self.fsm_state
    }

    pub fn max_history_size(&self) -> usize {
        self.max_history_size
    }

    pub fn current_order(&self) -> usize {
        self.current_order
    }

    pub fn at_downsample(&self) -> bool {
        self.at_downsample
    }

    pub fn at_reinitialize(&self) -> bool {
        self.at_reinitialize
    }

    pub fn at_order_change(&self) -> bool {
        self.at_order_change
    }

    pub fn at_end_of_tour(&self) -> bool {
        self.at_end_of_tour
    }

    /// JEOD: `GaussJacksonStateMachine::get_cycle_scale()`.
    pub fn cycle_scale(&self) -> f64 {
        self.cycle_scale
    }

    #[allow(dead_code)]
    pub fn history_length(&self) -> usize {
        self.history_length
    }

    // ── Mutators ──

    /// Tell the state machine that the edit did not pass convergence.
    /// Only requests a redo if another iteration is still allowed; otherwise
    /// no redo is requested and the edit proceeds with the non-converged
    /// result.
    ///
    /// JEOD: `GaussJacksonStateMachine::set_bootstrap_edit_redo_needed()`.
    pub fn set_bootstrap_edit_redo_needed(&mut self) {
        assert_eq!(self.fsm_state, FsmState::BootstrapEdit);
        if self.correction_iterations < self.max_correction_iterations {
            self.bootstrap_edit_redo_needed = true;
        }
    }

    /// Reset the state machine.
    ///
    /// JEOD: `GaussJacksonStateMachine::reset()`.
    pub fn reset(&mut self) {
        self.fsm_state = FsmState::Reset;
        self.history_length = 0;
        self.history_size = 1;

        self.step_increment = 1;
        self.steps_since_reset = 0;

        self.current_order = self.initial_order;

        self.scale_factor = self.tour_count;
        self.cycle_scale = 1.0 / self.tour_count as f64;
        self.cycle_start_time = 0.0;

        self.at_downsample = false;
        self.at_reinitialize = false;
        self.at_order_change = false;
        self.at_end_of_tour = false;
    }

    /// Advance the state machine by one step.
    ///
    /// JEOD: `GaussJacksonStateMachine::perform_step()`.
    pub fn perform_step(&mut self) {
        self.at_downsample = false;
        self.at_reinitialize = false;
        self.at_order_change = false;
        self.at_end_of_tour = false;

        self.history_length += 1;

        if self.history_length == self.history_size {
            self.transition_state();
        }

        if self.fsm_state == FsmState::BootstrapEdit {
            self.steps_since_reset = self.history_length * self.step_increment;
            self.cycle_start_time = self.steps_since_reset as f64 / self.tour_count as f64;
        } else {
            self.cycle_start_time = self.steps_since_reset as f64 / self.tour_count as f64;
            self.steps_since_reset += self.step_increment;
            self.at_end_of_tour = self.steps_since_reset.is_multiple_of(self.tour_count);
        }
    }

    /// State transition dispatch.
    ///
    /// JEOD: `GaussJacksonStateMachine::transition_state()`.
    fn transition_state(&mut self) {
        match self.fsm_state {
            FsmState::Reset => {
                // Reset → Priming
                self.fsm_state = FsmState::Priming;
                self.current_order = 0;
                self.history_size = self.initial_order + 1;
                self.steps_since_reset = 0;
            }
            FsmState::Priming => {
                self.exit_priming();
            }
            FsmState::BootstrapEdit => {
                self.exit_bootstrap_edit();
            }
            FsmState::BootstrapStep => {
                self.exit_bootstrap_step();
            }
            FsmState::Operational => {
                // No transition — operational is terminal
            }
        }
    }

    /// Transition out of Priming.
    ///
    /// JEOD: `GaussJacksonStateMachine::exit_priming()`.
    fn exit_priming(&mut self) {
        self.current_order = self.initial_order;
        self.at_order_change = true;
        self.at_reinitialize = true;

        if self.max_correction_iterations > 0 {
            self.bootstrap_edit_redo_needed = false;
            self.history_length = 1;
            self.correction_iterations = 1;
            self.fsm_state = FsmState::BootstrapEdit;
        } else {
            self.bootstrap_edit_redo_needed = false;
            self.exit_bootstrap_edit();
        }
    }

    /// Transition out of BootstrapEdit.
    ///
    /// JEOD: `GaussJacksonStateMachine::exit_bootstrap_edit()`.
    fn exit_bootstrap_edit(&mut self) {
        self.at_reinitialize = true;

        if self.bootstrap_edit_redo_needed {
            // Edit failed: redo with new derivatives
            self.bootstrap_edit_redo_needed = false;
            self.history_length = 1;
            self.correction_iterations += 1;
            self.fsm_state = FsmState::BootstrapEdit;
        } else if self.scale_factor == 1 && self.current_order == self.final_order {
            // At final order and step size → Operational
            self.fsm_state = FsmState::Operational;
        } else {
            // Need more bootstrapping
            self.fsm_state = FsmState::BootstrapStep;

            if self.scale_factor == 1 {
                // Change order only
                self.history_size = (self.current_order + 2) + 1;
            } else if self.current_order == self.final_order {
                // Downsample only
                self.history_size = 2 * self.current_order + 1;
            } else {
                // Both downsample and change order
                self.history_size = 2 * (self.current_order + 2) + 1;
            }
        }
    }

    /// Transition out of BootstrapStep.
    ///
    /// JEOD: `GaussJacksonStateMachine::exit_bootstrap_step()`.
    fn exit_bootstrap_step(&mut self) {
        self.at_reinitialize = true;

        // Downsample if not at final step size
        if self.scale_factor != 1 {
            self.at_downsample = true;
            self.history_size = self.history_size.div_ceil(2);
            self.scale_factor /= 2;
            self.step_increment *= 2;
            self.cycle_scale *= 2.0;
        }

        // Increase order if not at final order
        if self.current_order != self.final_order {
            self.at_order_change = true;
            self.current_order += 2;
        }

        // At least one of above must be true
        assert!(self.at_downsample || self.at_order_change);

        if self.max_correction_iterations > 0 {
            self.bootstrap_edit_redo_needed = false;
            self.history_length = 1;
            self.correction_iterations = 1;
            self.fsm_state = FsmState::BootstrapEdit;
        } else {
            self.bootstrap_edit_redo_needed = false;
            self.exit_bootstrap_edit();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_order_path() {
        // With initial = final = 4, ndoubling = 0:
        // Reset → Priming → BootstrapEdit → Operational
        let config = GaussJacksonConfig::with_order(4);
        let mut sm = StateMachine::configure(&config);
        assert_eq!(sm.fsm_state(), FsmState::Reset);

        // Step 1: Reset → Priming (history_size = 1, triggers immediately)
        sm.perform_step();
        assert_eq!(sm.fsm_state(), FsmState::Priming);

        // Steps 2-5: Priming (history_size = order+1 = 5)
        for _ in 0..3 {
            sm.perform_step();
        }
        assert_eq!(sm.fsm_state(), FsmState::Priming);
        sm.perform_step(); // Step 5: history_length=5 == history_size → BootstrapEdit
        assert_eq!(sm.fsm_state(), FsmState::BootstrapEdit);

        // BootstrapEdit: edit order=4 points (history_size = order+1 = 5)
        // SM resets history_length to 1, needs 4 more steps to reach 5.
        for _ in 0..3 {
            sm.perform_step();
        }
        assert_eq!(sm.fsm_state(), FsmState::BootstrapEdit);
        sm.perform_step(); // history_length=5 == history_size → Operational
        assert_eq!(sm.fsm_state(), FsmState::Operational);
    }

    #[test]
    fn test_bootstrap_path() {
        // With initial = 4, final = 8, ndoubling = 2:
        // Needs BootstrapStep phases
        let config = GaussJacksonConfig {
            initial_order: 4,
            final_order: 8,
            ndoubling_steps: 2,
            ..Default::default()
        };
        let mut sm = StateMachine::configure(&config);
        assert_eq!(sm.fsm_state(), FsmState::Reset);

        // Run until operational (should eventually get there)
        for _ in 0..200 {
            sm.perform_step();
            if sm.fsm_state() == FsmState::Operational {
                break;
            }
        }
        assert_eq!(sm.fsm_state(), FsmState::Operational);
    }
}
