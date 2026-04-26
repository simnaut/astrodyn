//! Time-stamped mass-tree event schedule for attach / detach tests.
//!
//! Phase 8 of #101: lifted from inlined `match step { ... }` patterns
//! in the `tier3_sim_attach*.rs` family. The schedule lets a bespoke
//! propagation loop ask "should I apply an event at time `t`?" without
//! sprinkling step-count arithmetic through the test body.

/// A single mass-tree event. The tag is opaque — tests interpret it
/// through their own match.
#[derive(Debug, Clone, Copy)]
pub struct AttachDetachEvent<E> {
    /// Simulation time (s) at which the event fires.
    pub time: f64,
    /// Test-specific event tag (typically `Attach` / `Detach`).
    pub event: E,
}

impl<E> AttachDetachEvent<E> {
    pub fn new(time: f64, event: E) -> Self {
        Self { time, event }
    }
}

/// Sorted-by-time queue of mass-tree events.
///
/// Tests typically build the schedule once at setup and call
/// [`take_due`](Self::take_due) each step with the current sim time.
#[derive(Debug, Clone, Default)]
pub struct EventSchedule<E> {
    pending: std::collections::VecDeque<AttachDetachEvent<E>>,
}

impl<E: Copy> EventSchedule<E> {
    /// Build a schedule from a slice; sorts by ascending time.
    pub fn new(events: &[AttachDetachEvent<E>]) -> Self {
        let mut v: Vec<_> = events.to_vec();
        v.sort_by(|a, b| {
            a.time
                .partial_cmp(&b.time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Self {
            pending: v.into_iter().collect(),
        }
    }

    /// Pop and return all events whose `time <= now`.
    pub fn take_due(&mut self, now: f64) -> Vec<AttachDetachEvent<E>> {
        let mut due = Vec::new();
        while let Some(front) = self.pending.front() {
            if front.time <= now {
                due.push(self.pending.pop_front().unwrap());
            } else {
                break;
            }
        }
        due
    }

    /// Number of events not yet fired.
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Returns `true` once every event has fired.
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Ev {
        Attach,
        Detach,
    }

    #[test]
    fn fires_in_time_order() {
        let mut sched = EventSchedule::new(&[
            AttachDetachEvent::new(20.0, Ev::Detach),
            AttachDetachEvent::new(10.0, Ev::Attach),
        ]);
        assert_eq!(sched.pending_len(), 2);
        let due = sched.take_due(15.0);
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].event, Ev::Attach);
        let due = sched.take_due(25.0);
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].event, Ev::Detach);
        assert!(sched.is_empty());
    }
}
