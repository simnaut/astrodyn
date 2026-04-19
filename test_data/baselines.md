# Tier 3 Baselines (frozen)

Per-test, per-component max absolute errors captured at the Phase 0 freeze point of
GitHub issue #101. See `CLAUDE.md` §"Baseline freeze" for the invariance policy.

76 tests recorded.

## `tier3_apollo8_eci_integ`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 4.518032e-5 | 3.631413e-5 | 3.398955e-5 | m |
| velocity | 9.124617e-7 | 7.291409e-7 | 6.806365e-7 | m/s |
| quat_angle | 0e0 |  |  | rad |
| ang_vel | 0e0 | 0e0 | 0e0 | rad/s |

## `tier3_apollo_mass_tree`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| composite_mass | 4.898757e-7 |  |  | kg |
| composite_com | 4.586072e-7 |  |  | m |
| composite_inertia | 5.960464e-7 |  |  | kg*m^2 |

## `tier3_earth_moon_clem`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 7.92267e-1 | 3.152603e-1 | 9.254903e-1 | m |
| velocity | 4.118806e-4 | 1.236043e-4 | 3.790989e-4 | m/s |

## `tier3_mars_dawn`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 3.811196e0 | 8.754798e-1 | 1.244367e0 | m |
| velocity | 8.134289e-4 | 4.211432e-4 | 5.027424e-4 | m/s |

## `tier3_planetary_geo`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 8.046627e-7 | 4.582107e-7 | 6.266418e-10 | m |
| velocity | 3.342393e-11 | 5.724132e-11 | 3.175238e-14 | m/s |

## `tier3_planetary_leo_ecc`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 1.091705e-5 | 8.541712e-6 | 9.307527e-9 | m |
| velocity | 7.587914e-9 | 1.053672e-8 | 6.608936e-12 | m/s |

## `tier3_planetary_leo_equ`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 2.05629e-5 | 2.069361e-5 | 1.733207e-8 | m |
| velocity | 2.332455e-8 | 2.341308e-8 | 1.965068e-11 | m/s |

## `tier3_planetary_leo_inc`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 3.599271e-6 | 4.908972e-6 | 3.539205e-6 | m |
| velocity | 3.973184e-9 | 5.529728e-9 | 3.942802e-9 | m/s |

## `tier3_planetary_leo_polar`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 3.298395e-6 | 1.81946e-5 | 1.872642e-5 | m |
| velocity | 3.742088e-9 | 2.064125e-8 | 2.089516e-8 | m/s |

## `tier3_reference_run10a_libration_period`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| period_error_pct | 3.73657e-1 |  |  | % |

## `tier3_sim_attach_detach_simple`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| composite_mass_max_err | 0e0 |  |  | kg |

## `tier3_sim_attach_mass`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| composite_mass | 0e0 |  |  | kg |
| composite_com | 4.082156e-17 |  |  | m |
| composite_inertia | 4e-7 |  |  | kg*m^2 |

## `tier3_sim_drag_ver_bc`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| aero_force_err | 2.637978e-18 |  |  | N |
| aero_torque_err | 0e0 |  |  | N*m |
| accel_mag_err | 2.602085e-18 |  |  | m/s^2 |

## `tier3_sim_drag_ver_cd`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| aero_force_err | 2.168404e-18 |  |  | N |
| aero_torque_err | 0e0 |  |  | N*m |
| accel_mag_err | 1.734723e-18 |  |  | m/s^2 |

## `tier3_sim_drag_ver_flatplate_calc_eps00`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| aero_force_err | 2.056959e-12 |  |  | N |
| aero_torque_err | 0e0 |  |  | N*m |
| accel_mag_err | 2.056451e-12 |  |  | m/s^2 |

## `tier3_sim_drag_ver_flatplate_calc_eps05`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| aero_force_err | 1.3503e-12 |  |  | N |
| aero_torque_err | 0e0 |  |  | N*m |
| accel_mag_err | 1.349134e-12 |  |  | m/s^2 |

## `tier3_sim_drag_ver_flatplate_calc_eps1`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| aero_force_err | 1.237368e-12 |  |  | N |
| aero_torque_err | 0e0 |  |  | N*m |
| accel_mag_err | 1.237368e-12 |  |  | m/s^2 |

## `tier3_sim_drag_ver_flatplate_diffuse`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| aero_force_err | 2.142843e-20 |  |  | N |
| aero_torque_err | 0e0 |  |  | N*m |
| accel_mag_err | 2.032879e-20 |  |  | m/s^2 |

## `tier3_sim_drag_ver_flatplate_mixed`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| aero_force_err | 2.793925e-20 |  |  | N |
| aero_torque_err | 0e0 |  |  | N*m |
| accel_mag_err | 4.065758e-20 |  |  | m/s^2 |

## `tier3_sim_drag_ver_flatplate_orbiter`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| aero_force_err | 9.347926e-10 |  |  | N |
| aero_torque_err | 1.180006e-9 |  |  | N*m |
| accel_mag_err | 1.020617e-14 |  |  | m/s^2 |

## `tier3_sim_drag_ver_flatplate_specular`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| aero_force_err | 6.776264e-20 |  |  | N |
| aero_torque_err | 0e0 |  |  | N*m |
| accel_mag_err | 6.776264e-20 |  |  | m/s^2 |

## `tier3_sim_drag_ver_flatplate_torque`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| aero_force_err | 2.056959e-12 |  |  | N |
| aero_torque_err | 6.186842e-13 |  |  | N*m |
| accel_mag_err | 2.056451e-12 |  |  | m/s^2 |

## `tier3_simulation_drag_run6b`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 7.590876e-1 | 1.060947e0 | 8.518602e-1 | m |
| velocity | 7.486293e-4 | 1.194132e-3 | 9.544923e-4 | m/s |

## `tier3_simulation_drag_run6b_rotated`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 7.590876e-1 | 1.060947e0 | 8.518602e-1 | m |
| velocity | 7.486293e-4 | 1.194132e-3 | 9.544923e-4 | m/s |

## `tier3_simulation_euler`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| quat_angle | 4.214685e-8 |  |  | rad |
| ang_vel | 2.493665e-18 | 1.301043e-18 | 7.589415e-19 | rad/s |
| euler_roll | 1.758038e-13 |  |  | rad |
| euler_pitch | 8.260059e-14 |  |  | rad |
| euler_yaw | 1.050271e-13 |  |  | rad |

## `tier3_simulation_euler_ecc`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| quat_angle | 0e0 |  |  | rad |
| euler_roll | 0e0 |  |  | rad |
| euler_pitch | 0e0 |  |  | rad |
| euler_yaw | 0e0 |  |  | rad |

## `tier3_simulation_euler_equ`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| quat_angle | 0e0 |  |  | rad |
| euler_roll | 0e0 |  |  | rad |
| euler_pitch | 0e0 |  |  | rad |
| euler_yaw | 0e0 |  |  | rad |

## `tier3_simulation_geodetic`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 3.599271e-6 | 4.908972e-6 | 3.539205e-6 | m |
| velocity | 3.973184e-9 | 5.529728e-9 | 3.942802e-9 | m/s |
| altitude | 8.51233e-4 |  |  | m |
| latitude | 3.982369e-8 |  |  | rad |
| longitude | 6.183679e-8 |  |  | rad |

## `tier3_simulation_gj_dt10`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 9.861992e-1 | 9.845695e-1 | 0e0 | m |
| velocity | 8.751316e-4 | 8.75485e-4 | 0e0 | m/s |

## `tier3_simulation_gj_order12`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 1.850567e-4 | 1.846611e-4 | 0e0 | m |
| velocity | 1.642576e-7 | 1.645428e-7 | 0e0 | m/s |

## `tier3_simulation_gj_order4`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 3.676175e-5 | 3.714341e-5 | 0e0 | m |
| velocity | 3.282528e-8 | 3.270148e-8 | 0e0 | m/s |

## `tier3_simulation_gj_order8`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 1.258367e-4 | 1.246376e-4 | 0e0 | m |
| velocity | 1.105691e-7 | 1.111997e-7 | 0e0 | m/s |

## `tier3_simulation_lsode_abm4`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 3.4055e-4 | 3.25971e-4 | 2.151958e-4 | m |
| velocity | 3.870171e-7 | 3.553018e-7 | 2.361918e-7 | m/s |

## `tier3_simulation_lsode_default`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 9.484859e3 | 9.129797e3 | 6.02849e3 | m |
| velocity | 1.082078e1 | 9.908454e0 | 6.580769e0 | m/s |

## `tier3_simulation_lvlh`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 3.599271e-6 | 4.908972e-6 | 3.539205e-6 | m |
| velocity | 3.973184e-9 | 5.529728e-9 | 3.942802e-9 | m/s |
| t_parent_this | 7.303186e-13 |  |  |  |
| ang_vel | 1.11456e-16 |  |  | rad/s |

## `tier3_simulation_lvlh_ecc`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 1.091705e-5 | 8.541712e-6 | 9.307527e-9 | m |
| velocity | 7.587914e-9 | 1.053672e-8 | 6.608936e-12 | m/s |
| t_parent_this | 1.608637e-12 |  |  |  |
| ang_vel | 6.542076e-16 |  |  | rad/s |

## `tier3_simulation_lvlh_equ`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 2.05629e-5 | 2.069361e-5 | 1.733207e-8 | m |
| velocity | 2.332455e-8 | 2.341308e-8 | 1.965068e-11 | m/s |
| t_parent_this | 3.052998e-12 |  |  |  |
| ang_vel | 1.400789e-16 |  |  | rad/s |

## `tier3_simulation_met_run5a`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 5.117618e-7 | 7.976778e-7 | 6.016344e-7 | m |
| velocity | 4.931735e-10 | 8.867573e-10 | 7.00993e-10 | m/s |

## `tier3_simulation_ned_polar`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 3.298395e-6 | 1.81946e-5 | 1.872642e-5 | m |
| velocity | 3.742088e-9 | 2.064125e-8 | 2.089516e-8 | m/s |
| altitude | 2.021248e-4 |  |  | m |
| latitude | 1.0362e-8 |  |  | rad |
| longitude | 3.189424e-5 |  |  | rad |

## `tier3_simulation_ned_sph_inc`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 3.599271e-6 | 4.908972e-6 | 3.539205e-6 | m |
| velocity | 3.973184e-9 | 5.529728e-9 | 3.942802e-9 | m/s |
| altitude | 3.828318e-7 |  |  | m |
| latitude | 3.981793e-8 |  |  | rad |
| longitude | 6.183679e-8 |  |  | rad |

## `tier3_simulation_ned_sph_polar`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 3.298395e-6 | 1.81946e-5 | 1.872642e-5 | m |
| velocity | 3.742088e-9 | 2.064125e-8 | 2.089516e-8 | m/s |
| altitude | 3.793393e-7 |  |  | m |
| latitude | 1.031194e-8 |  |  | rad |
| longitude | 3.189424e-5 |  |  | rad |

## `tier3_simulation_orbelem`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 1.091705e-5 | 8.541712e-6 | 9.307527e-9 | m |
| velocity | 7.587914e-9 | 1.053672e-8 | 6.608936e-12 | m/s |
| sma | 5.047768e-7 |  |  | m |
| eccentricity | 2.09277e-14 |  |  |  |
| inclination | 3.22008e-17 |  |  | rad |
| arg_periapsis | 4.013456e-14 |  |  | rad |
| long_asc_node | 2.220446e-14 |  |  | rad |
| true_anom | 1.638689e-12 |  |  | rad |
| mean_anom | 7.400747e-13 |  |  | rad |

## `tier3_simulation_run10a_gravity_torque`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 1.304084e-6 | 2.050656e-6 | 1.738284e-6 | m |
| velocity | 1.377089e-9 | 2.274305e-9 | 1.726789e-9 | m/s |
| quat_angle | 7.195352e-5 |  |  | rad |
| ang_vel | 0e0 | 1.11569e-7 | 8.858036e-8 | rad/s |

## `tier3_simulation_run10c_gravity_torque_elliptical`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 5.117618e-7 | 7.976778e-7 | 6.016344e-7 | m |
| velocity | 4.931735e-10 | 8.867573e-10 | 7.00993e-10 | m/s |
| quat_angle | 7.597446e-5 |  |  | rad |
| ang_vel | 0e0 | 1.183046e-7 | 9.186244e-8 | rad/s |

## `tier3_simulation_run10d_gravity_torque_elliptical_rate`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 5.117618e-7 | 7.976778e-7 | 6.016344e-7 | m |
| velocity | 4.931735e-10 | 8.867573e-10 | 7.00993e-10 | m/s |
| quat_angle | 1.052975e-4 |  |  | rad |
| ang_vel | 0e0 | 1.7377e-7 | 1.138897e-7 | rad/s |

## `tier3_simulation_run2_3dof`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 1.304084e-6 | 2.050656e-6 | 1.738284e-6 | m |
| velocity | 1.377089e-9 | 2.274305e-9 | 1.726789e-9 | m/s |

## `tier3_simulation_run2_6dof`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 1.304084e-6 | 2.050656e-6 | 1.738284e-6 | m |
| velocity | 1.377089e-9 | 2.274305e-9 | 1.726789e-9 | m/s |
| quat_angle | 4.214685e-8 |  |  | rad |
| ang_vel | 2.493665e-18 | 1.301043e-18 | 7.589415e-19 | rad/s |

## `tier3_simulation_run2p_polar_motion`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 1.304084e-6 | 2.050656e-6 | 1.738284e-6 | m |
| velocity | 1.377089e-9 | 2.274305e-9 | 1.726789e-9 | m/s |

## `tier3_simulation_run3a_sh4x4`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 5.046699e-2 | 1.279437e-1 | 9.762894e-2 | m |
| velocity | 5.857447e-5 | 1.186332e-4 | 1.180319e-4 | m/s |

## `tier3_simulation_run3b_sh8x8`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 1.261038e-1 | 2.190092e-1 | 1.567328e-1 | m |
| velocity | 1.407207e-4 | 2.21787e-4 | 1.801017e-4 | m/s |

## `tier3_simulation_run4_3rd_body`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 1.565674e-3 | 1.998357e-3 | 1.929048e-3 | m |
| velocity | 1.678174e-6 | 1.982956e-6 | 2.286183e-6 | m/s |
| quat_angle | 4.214685e-8 |  |  | rad |
| ang_vel | 2.493665e-18 | 1.301043e-18 | 7.589415e-19 | rad/s |

## `tier3_simulation_run5b_atmosphere_mean`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 5.117618e-7 | 7.976778e-7 | 6.016344e-7 | m |
| velocity | 4.931735e-10 | 8.867573e-10 | 7.00993e-10 | m/s |
| quat_angle | 4.214685e-8 |  |  | rad |
| ang_vel | 3.361027e-18 | 1.626303e-18 | 1.057097e-18 | rad/s |

## `tier3_simulation_run5c_atmosphere_max`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 5.117618e-7 | 7.976778e-7 | 6.016344e-7 | m |
| velocity | 4.931735e-10 | 8.867573e-10 | 7.00993e-10 | m/s |
| quat_angle | 4.214685e-8 |  |  | rad |
| ang_vel | 3.361027e-18 | 1.626303e-18 | 1.057097e-18 | rad/s |

## `tier3_simulation_run6a_const_density_drag`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 4.157808e-4 | 6.5137e-4 | 5.070973e-4 | m |
| velocity | 4.70596e-7 | 7.124963e-7 | 5.861554e-7 | m/s |
| quat_angle | 4.214685e-8 |  |  | rad |
| ang_vel | 0e0 | 0e0 | 0e0 | rad/s |

## `tier3_simulation_run6b_drag`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 7.590876e-1 | 1.060947e0 | 8.518602e-1 | m |
| velocity | 7.486293e-4 | 1.194132e-3 | 9.544923e-4 | m/s |
| quat_angle | 4.214685e-8 |  |  | rad |
| ang_vel | 0e0 | 0e0 | 0e0 | rad/s |

## `tier3_simulation_run7a`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 4.885537e-2 | 1.253304e-1 | 9.509643e-2 | m |
| velocity | 5.75278e-5 | 1.148362e-4 | 1.15966e-4 | m/s |

## `tier3_simulation_run7b`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 1.219087e-1 | 2.142055e-1 | 1.520824e-1 | m |
| velocity | 1.377426e-4 | 2.142579e-4 | 1.767293e-4 | m/s |

## `tier3_simulation_run7c`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 6.654953e-1 | 9.882598e-1 | 8.117137e-1 | m |
| velocity | 6.723547e-4 | 1.10034e-3 | 9.109266e-4 | m/s |

## `tier3_simulation_run7d`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 7.366604e-1 | 1.072298e0 | 8.682969e-1 | m |
| velocity | 7.521539e-4 | 1.198347e-3 | 9.805374e-4 | m/s |

## `tier3_simulation_run9a_torque`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 1.304084e-6 | 2.050656e-6 | 1.738284e-6 | m |
| velocity | 1.377089e-9 | 2.274305e-9 | 1.726789e-9 | m/s |
| quat_angle | 4.214685e-8 |  |  | rad |
| ang_vel | 3.388132e-20 | 4.235165e-21 | 6.776264e-21 | rad/s |

## `tier3_simulation_run9c_force_torque`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 7.444271e-5 | 1.162612e-4 | 8.387904e-5 | m |
| velocity | 8.28056e-8 | 1.228855e-7 | 1.029078e-7 | m/s |
| quat_angle | 4.214685e-8 |  |  | rad |
| ang_vel | 3.388132e-20 | 4.235165e-21 | 6.776264e-21 | rad/s |

## `tier3_simulation_run9d_force_torque_rate`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 5.026444e-3 | 7.861266e-3 | 6.318404e-3 | m |
| velocity | 5.629213e-6 | 8.62444e-6 | 6.928789e-6 | m/s |
| quat_angle | 4.214685e-8 |  |  | rad |
| ang_vel | 1.572093e-18 | 1.301043e-18 | 5.963112e-19 | rad/s |

## `tier3_simulation_shadow_2a_annular`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| shadow_fraction | 3.819018e-5 |  |  |  |
| shadow_mismatches | 0e0 |  |  |  |

## `tier3_simulation_shadow_2a_cooling`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| shadow_fraction | 0e0 |  |  |  |
| shadow_mismatches | 0e0 |  |  |  |

## `tier3_simulation_solar_beta`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 1.304084e-6 | 2.050656e-6 | 1.738284e-6 | m |
| velocity | 1.377089e-9 | 2.274305e-9 | 1.726789e-9 | m/s |

## `tier3_simulation_solar_beta_equ`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 1.279433e-4 | 1.258097e-4 | 0e0 | m |
| velocity | 1.423099e-7 | 1.44125e-7 | 0e0 | m/s |
| beta | 1.801512e-5 |  |  | rad |

## `tier3_simulation_solar_beta_obliquity`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 9.365415e-1 | 8.146839e-1 | 3.233321e-1 | m |
| velocity | 9.953047e-4 | 9.825667e-4 | 3.922077e-4 | m/s |
| beta | 3.28184e-5 |  |  | rad |

## `tier3_simulation_srp_flat_plate`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 3.25432e-2 | 3.746371e-2 | 1.503887e-2 | m |
| velocity | 2.083585e-6 | 1.878458e-6 | 7.264819e-7 | m/s |

## `tier3_simulation_tide_run01`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 6.492773e-3 | 9.850911e-3 | 6.71119e-3 | m |
| velocity | 7.679143e-6 | 1.021112e-5 | 7.900142e-6 | m/s |

## `tier3_srp_1st_order_trajectory`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 7.336668e1 | 7.634384e1 | 3.313474e1 | m |
| velocity | 5.640582e-3 | 4.924875e-3 | 2.136479e-3 | m/s |

## `tier3_torque_simple_run01`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 5.097594e-6 | 8.704606e-6 | 9.525102e-6 | m |
| velocity | 8.738425e-9 | 7.504696e-9 | 9.101313e-9 | m/s |
| quat_angle | 3.141472e0 |  |  | rad |
| ang_vel | 2.140038e-3 | 2.985749e-3 | 4.760225e-4 | rad/s |
| torque | 0e0 |  |  | N*m |

## `tier3_torque_simple_run02`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 5.097594e-6 | 8.704606e-6 | 9.525102e-6 | m |
| velocity | 8.738425e-9 | 7.504696e-9 | 9.101313e-9 | m/s |
| quat_angle | 3.575479e-2 |  |  | rad |
| ang_vel | 4.085699e-5 | 3.078332e-5 | 2.560861e-6 | rad/s |
| torque | 0e0 |  |  | N*m |

## `tier3_torque_simple_run03`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 5.097594e-6 | 8.704606e-6 | 9.525102e-6 | m |
| velocity | 8.738425e-9 | 7.504696e-9 | 9.101313e-9 | m/s |
| quat_angle | 3.575479e-2 |  |  | rad |
| ang_vel | 4.085699e-5 | 3.078332e-5 | 2.560861e-6 | rad/s |
| torque | 0e0 |  |  | N*m |

## `tier3_torque_simple_run04`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 2.939746e-1 | 4.612822e-1 | 4.062317e-1 | m |
| velocity | 3.380864e-4 | 5.330711e-4 | 3.914765e-4 | m/s |
| quat_angle | 3.141174e0 |  |  | rad |
| ang_vel | 2.136765e-3 | 3.034934e-3 | 4.739924e-4 | rad/s |
| torque | 0e0 |  |  | N*m |

## `tier3_torque_simple_run05`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 2.939746e-1 | 4.612822e-1 | 4.062317e-1 | m |
| velocity | 3.380864e-4 | 5.330711e-4 | 3.914765e-4 | m/s |
| quat_angle | 1.723056e-2 |  |  | rad |
| ang_vel | 1.719936e-5 | 1.344094e-5 | 4.278566e-6 | rad/s |
| torque | 0e0 |  |  | N*m |

## `tier3_torque_simple_run06`

| Metric | X / value | Y | Z | Unit |
|--------|-----------|---|---|------|
| position | 2.939746e-1 | 4.612822e-1 | 4.062317e-1 | m |
| velocity | 3.380864e-4 | 5.330711e-4 | 3.914765e-4 | m/s |
| quat_angle | 5.942505e-1 |  |  | rad |
| ang_vel | 5.424455e-4 | 4.806425e-4 | 1.665372e-4 | rad/s |
| torque | 0e0 |  |  | N*m |

