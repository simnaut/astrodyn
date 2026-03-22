// Standalone JEOD gravity benchmark — computes point-mass gravity at test
// positions from verif_out.txt and outputs CSV for cross-validation with
// bevy_jeod. Does NOT require Trick or the JEOD test harness.
//
// Build:
//   g++ -O2 -std=c++11 -I${JEOD_HOME}/models \
//       -o jeod_gravity_benchmark jeod_gravity_benchmark.cc \
//       -L${JEOD_HOME}/build -ljeod -lm
//
// Run:
//   ./jeod_gravity_benchmark < ${JEOD_HOME}/models/environment/gravity/verif/unit_tests/grav_geospherical/data/verif_out.txt

#include <cstdio>
#include <cmath>
#include <cstdlib>

// JEOD's GM for Earth (from gravity data)
static const double MU_EARTH = 3.986004418e14;  // m^3/s^2

// Simple point-mass gravity computation (matching JEOD's calc_spherical)
void compute_point_mass(const double pos[3], double mu,
                        double accel[3], double *potential,
                        double grad[3][3]) {
    double r_sq = pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2];
    double r_mag = sqrt(r_sq);
    double r_3rd = r_sq * r_mag;
    double r_5th = r_3rd * r_sq;

    double mu_r3 = mu / r_3rd;
    double mu_r5 = mu / r_5th;

    // Acceleration: a = -mu/r^3 * r
    for (int i = 0; i < 3; i++) {
        accel[i] = -mu_r3 * pos[i];
    }

    // Potential: V = -mu/r
    *potential = -mu / r_mag;

    // Gradient: G[i][j] = mu/r^5 * (3*r_i*r_j - delta_ij*r^2)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            grad[i][j] = mu_r5 * (3.0 * pos[i] * pos[j]);
            if (i == j) {
                grad[i][j] -= mu_r5 * r_sq;
            }
        }
    }
}

int main() {
    // Read verif_out.txt from stdin, compute point-mass gravity, output CSV
    printf("case,degree,order,");
    printf("pos_x,pos_y,pos_z,");
    printf("jeod_accel_x,jeod_accel_y,jeod_accel_z,");
    printf("jeod_potential,");
    printf("pm_accel_x,pm_accel_y,pm_accel_z,");
    printf("pm_potential,");
    printf("accel_diff_mag,pot_diff\n");

    char line[4096];
    while (fgets(line, sizeof(line), stdin)) {
        int caseNum, degree, order, perturbOnly, gradActive;
        double pos[3], potential, accel[3];
        double dgdx[6]; // upper triangle

        int n = sscanf(line,
            "%d %d %d %d %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
            &caseNum, &degree, &order, &perturbOnly, &gradActive,
            &pos[0], &pos[1], &pos[2],
            &potential,
            &accel[0], &accel[1], &accel[2],
            &dgdx[0], &dgdx[1], &dgdx[2], &dgdx[3], &dgdx[4], &dgdx[5]);

        if (n < 12) continue;

        // Compute point-mass gravity at same position
        double pm_accel[3], pm_pot;
        double pm_grad[3][3];
        compute_point_mass(pos, MU_EARTH, pm_accel, &pm_pot, pm_grad);

        // Difference
        double diff_x = pm_accel[0] - accel[0];
        double diff_y = pm_accel[1] - accel[1];
        double diff_z = pm_accel[2] - accel[2];
        double diff_mag = sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z);
        double pot_diff = pm_pot - potential;

        printf("%d,%d,%d,", caseNum, degree, order);
        printf("%.15e,%.15e,%.15e,", pos[0], pos[1], pos[2]);
        printf("%.15e,%.15e,%.15e,", accel[0], accel[1], accel[2]);
        printf("%.15e,", potential);
        printf("%.15e,%.15e,%.15e,", pm_accel[0], pm_accel[1], pm_accel[2]);
        printf("%.15e,", pm_pot);
        printf("%.15e,%.15e\n", diff_mag, pot_diff);
    }

    return 0;
}
