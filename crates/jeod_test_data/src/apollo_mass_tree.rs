//! Parser for JEOD's `MassBody::print_body` output format.
//!
//! The `print_body` method (mass_print_body.cc) writes a hierarchical text
//! representation of the mass tree, one section per body. Each section has
//! a fixed format with labeled numeric fields that this module parses into
//! a structured [`PrintedTree`].
//!
//! Used by the Tier 3 Apollo mass tree test to load JEOD reference data.

use glam::{DMat3, DVec3};

/// One body's printed mass properties.
#[derive(Debug, Clone)]
pub struct PrintedBody {
    pub name: String,
    /// Structural offset in parent's structural frame (m).
    pub structure_offset: DVec3,
    /// Rotation from parent struct to this struct.
    pub structure_rotation: DMat3,
    /// Core center of mass in structural frame (m).
    pub core_cm: DVec3,
    /// Core mass (kg).
    pub core_mass: f64,
    /// Core inertia tensor about body frame axes through CoM (kg*m^2).
    pub core_inertia: DMat3,
    /// Composite center of mass in structural frame (m).
    pub composite_cm: DVec3,
    /// Composite mass (kg).
    pub composite_mass: f64,
    /// Composite inertia tensor (kg*m^2).
    pub composite_inertia: DMat3,
}

/// A complete mass tree printout (one or more bodies).
#[derive(Debug, Clone)]
pub struct PrintedTree {
    pub bodies: Vec<PrintedBody>,
}

impl PrintedTree {
    /// Find a body by name.
    pub fn find(&self, name: &str) -> Option<&PrintedBody> {
        self.bodies.iter().find(|b| b.name == name)
    }
}

/// Parse a JEOD `print_tree` output file into a [`PrintedTree`].
///
/// The file contains one or more body sections separated by `====` lines.
/// Each section has the format documented in `mass_print_body.cc`.
pub fn parse_print_tree(content: &str) -> PrintedTree {
    let mut bodies = Vec::new();
    let sections: Vec<&str> = content
        .split("=============================================================")
        .collect();

    for section in sections {
        let section = section.trim();
        if section.is_empty() {
            continue;
        }

        // First non-empty line is the body name.
        let lines: Vec<&str> = section.lines().collect();
        let mut i = 0;

        // Skip empty lines.
        while i < lines.len() && lines[i].trim().is_empty() {
            i += 1;
        }
        if i >= lines.len() {
            continue;
        }

        let name = lines[i].trim().to_string();
        i += 1;

        // Skip separator line.
        while i < lines.len() && lines[i].contains("-----") {
            i += 1;
        }

        // Parse "Body Area" section.
        skip_label(&lines, &mut i, "Body Area");
        skip_label(&lines, &mut i, "Offset Vector");
        let structure_offset = parse_vec3(&lines, &mut i);
        skip_label(&lines, &mut i, "T_struct_struct");
        let structure_rotation = parse_mat3(&lines, &mut i);

        // Skip separator.
        skip_separator(&lines, &mut i);

        // Parse "Mass Properties" section.
        skip_label(&lines, &mut i, "Mass Properties");
        skip_label(&lines, &mut i, "M.P. CM vector");
        let core_cm = parse_vec3(&lines, &mut i);
        skip_label(&lines, &mut i, "M.P. Mass");
        let core_mass = parse_f64(&lines, &mut i);
        skip_label(&lines, &mut i, "M.P. Ib tensor");
        let core_inertia = parse_mat3(&lines, &mut i);
        // Skip T_struct_body (we don't need it for composite validation).
        skip_label(&lines, &mut i, "M.P. T_struct_body");
        let _core_t_struct_body = parse_mat3(&lines, &mut i);

        // Skip separator.
        skip_separator(&lines, &mut i);

        // Parse "Composite Mass Properties" section.
        skip_label(&lines, &mut i, "Composite Mass Properties");
        skip_label(&lines, &mut i, "C.M.P. CM vector");
        let composite_cm = parse_vec3(&lines, &mut i);
        skip_label(&lines, &mut i, "C.M.P. Mass");
        let composite_mass = parse_f64(&lines, &mut i);
        skip_label(&lines, &mut i, "C.M.P. Ib tensor");
        let composite_inertia = parse_mat3(&lines, &mut i);

        // We don't parse "C.M.P. T_struct_body" or "Derived Items" — not needed.

        bodies.push(PrintedBody {
            name,
            structure_offset,
            structure_rotation,
            core_cm,
            core_mass,
            core_inertia,
            composite_cm,
            composite_mass,
            composite_inertia,
        });
    }

    PrintedTree { bodies }
}

/// Advance past a labeled line. Panics with line number on unexpected content.
/// For richer error context (body name, filename), callers should catch panics
/// or this should be refactored to return Result (future improvement).
fn skip_label(lines: &[&str], i: &mut usize, label: &str) {
    while *i < lines.len() {
        let line = lines[*i].trim();
        if line.contains(label) {
            *i += 1;
            return;
        }
        if line.is_empty() || line.contains("-----") || line.contains("====") {
            *i += 1;
            continue;
        }
        panic!(
            "expected label {:?} before data, found {:?} at line {}",
            label,
            line,
            *i + 1
        );
    }
    panic!("expected label {:?} before end of input", label);
}

fn skip_separator(lines: &[&str], i: &mut usize) {
    while *i < lines.len() && (lines[*i].contains("-----") || lines[*i].trim().is_empty()) {
        *i += 1;
    }
}

/// Parse a single f64 from the next line containing a number.
fn parse_f64(lines: &[&str], i: &mut usize) -> f64 {
    while *i < lines.len() {
        let line = lines[*i].trim();
        *i += 1;
        if line.is_empty() || line.contains("-----") {
            continue;
        }
        if let Some(val) = line.split_whitespace().find_map(|s| s.parse::<f64>().ok()) {
            return val;
        }
    }
    panic!("unexpected end of input while parsing f64");
}

/// Parse a DVec3 from the next line containing 3 whitespace-separated floats.
fn parse_vec3(lines: &[&str], i: &mut usize) -> DVec3 {
    while *i < lines.len() {
        let line = lines[*i].trim();
        *i += 1;
        if line.is_empty() || line.contains("-----") {
            continue;
        }
        // Try to parse all whitespace-separated tokens as f64.
        let vals: Vec<f64> = line
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        if vals.len() == 3 {
            return DVec3::new(vals[0], vals[1], vals[2]);
        }
        // Not a numeric line — skip (could be a label we didn't expect).
    }
    panic!("unexpected end of input while parsing vec3");
}

/// Parse a DMat3 from 3 consecutive lines (3 floats per row).
fn parse_mat3(lines: &[&str], i: &mut usize) -> DMat3 {
    let row0 = parse_vec3(lines, i);
    let row1 = parse_vec3(lines, i);
    let row2 = parse_vec3(lines, i);
    // JEOD prints row-major: row0 = [T[0][0], T[0][1], T[0][2]].
    // glam DMat3::from_cols takes column vectors.
    DMat3::from_cols(
        DVec3::new(row0.x, row1.x, row2.x),
        DVec3::new(row0.y, row1.y, row2.y),
        DVec3::new(row0.z, row1.z, row2.z),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_OUTPUT: &str = r#"


=============================================================

cm

-------------------------------------------------------------

Body Area
Offset Vector [m]:
             0.000000000000             0.000000000000             0.000000000000
T_struct_struct [-]:
             1.000000000000             0.000000000000             0.000000000000
             0.000000000000             1.000000000000             0.000000000000
             0.000000000000             0.000000000000             1.000000000000
-------------------------------------------------------------
Mass Properties
M.P. CM vector [m]:
             2.651760000000             0.000000000000             0.000000000000
M.P. Mass [kg]:
          5810.305590000000
M.P. Ib tensor [kgM2]:
          6630.553660000000             0.000000000000             0.000000000000
             0.000000000000          2722.695540000000             0.000000000000
             0.000000000000             0.000000000000          2722.695540000000
M.P. T_struct_body [q]:
            -1.000000000000             0.000000000000             0.000000000000
             0.000000000000            -1.000000000000             0.000000000000
             0.000000000000             0.000000000000             1.000000000000
-------------------------------------------------------------
Composite Mass Properties
C.M.P. CM vector [m]:
             2.651760000000             0.000000000000             0.000000000000
C.M.P. Mass [kg]:
          5810.305590000000
C.M.P. Ib tensor [kgM2]:
          6630.553660000000             0.000000000000             0.000000000000
             0.000000000000          2722.695540000000             0.000000000000
             0.000000000000             0.000000000000          2722.695540000000
C.M.P. T_struct_body [q]:
            -1.000000000000             0.000000000000             0.000000000000
             0.000000000000            -1.000000000000             0.000000000000
             0.000000000000             0.000000000000             1.000000000000
-------------------------------------------------------------
Derived Items
C.M.P. Inverse mass [1/kg]:
             0.000172108000
C.M.P. Inverse inertia tensor [1/(kgM2)]:
             0.000150828000             0.000000000000             0.000000000000
             0.000000000000             0.000367278000             0.000000000000
             0.000000000000             0.000000000000             0.000367278000
-------------------------------------------------------------
"#;

    #[test]
    fn parse_single_body() {
        let tree = parse_print_tree(SAMPLE_OUTPUT);
        assert_eq!(tree.bodies.len(), 1);

        let cm = &tree.bodies[0];
        assert_eq!(cm.name, "cm");
        assert!((cm.core_mass - 5810.305590).abs() < 1e-4);
        assert!((cm.core_cm.x - 2.651760).abs() < 1e-6);
        assert!((cm.core_inertia.x_axis.x - 6630.553660).abs() < 1e-3);
        assert!((cm.composite_mass - 5810.305590).abs() < 1e-4);

        // Structure offset should be zero for root body.
        assert!(cm.structure_offset.length() < 1e-10);
        // Structure rotation should be identity.
        assert!((cm.structure_rotation - DMat3::IDENTITY).abs_diff_eq(DMat3::ZERO, 1e-10));
    }
}
