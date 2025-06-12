#include "LBMStreaming.h"
#include <iostream>
#include <fstream>
#include <iomanip>

LBMStreaming::LBMStreaming() :
    f("f", NX, NY, Q),
    f_new("f_new", NX, NY, Q),
    rho("rho", NX, NY),
    velocity("velocity", NX, NY, 2) {

    setup_velocity_vectors();
    initialize();
}

void LBMStreaming::setup_velocity_vectors() {
    // D2Q9 velocity vectors as shown in Figure 1a of the specification
    c[0][0] = 0;  c[0][1] = 0;   // stationary (center)
    c[1][0] = 1;  c[1][1] = 0;   // right
    c[2][0] = 0;  c[2][1] = 1;   // up
    c[3][0] = -1; c[3][1] = 0;   // left
    c[4][0] = 0;  c[4][1] = -1;  // down
    c[5][0] = 1;  c[5][1] = 1;   // up-right
    c[6][0] = -1; c[6][1] = 1;   // up-left
    c[7][0] = -1; c[7][1] = -1;  // down-left
    c[8][0] = 1;  c[8][1] = -1;  // down-right
}

void LBMStreaming::initialize() {
    // Initialize with simple patterns as suggested:
    // "start with values only in few (or even one) direction"
    // Think of f_i as "number of particles" at each position and direction

    Kokkos::parallel_for("initialize",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {NX, NY, Q}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            // Initialize all distributions to zero
            f(i, j, k) = 0.0;
            f_new(i, j, k) = 0.0;
        });

    // Add simple test pattern: particles at specific locations moving in specific directions
    Kokkos::parallel_for("set_test_pattern",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {NX, NY}),
        KOKKOS_LAMBDA(const int i, const int j) {
            // Test pattern 1: Particles at center moving right (direction 1)
            if (i >= 6 && i <= 8 && j >= 4 && j <= 5) {
                f(i, j, 1) = 1.0;  // 1 "particle" moving right
            }

            // Test pattern 2: Particles moving up-right (direction 5)
            if (i >= 3 && i <= 5 && j >= 2 && j <= 3) {
                f(i, j, 5) = 0.8;  // 0.8 "particles" moving up-right
            }

            // Test pattern 3: Some stationary particles (direction 0)
            if (i == 7 && j == 7) {
                f(i, j, 0) = 0.5;  // 0.5 "particles" stationary
            }
        });

    Kokkos::fence();
}

void LBMStreaming::compute_density() {
    // Density: ρ(x,y) = Σ f_i(x,y) - sum over all directions
    Kokkos::parallel_for("compute_density",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {NX, NY}),
        KOKKOS_LAMBDA(const int i, const int j) {
            double sum = 0.0;
            for (int k = 0; k < Q; k++) {
                sum += f(i, j, k);
            }
            rho(i, j) = sum;
        });

    Kokkos::fence();
}

void LBMStreaming::compute_velocity() {
    // Velocity: u(x,y) = (1/ρ) * Σ f_i(x,y) * c_i
    Kokkos::parallel_for("compute_velocity",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {NX, NY}),
        KOKKOS_LAMBDA(const int i, const int j) {
            double vx = 0.0, vy = 0.0;

            // Calculate momentum: Σ f_i * c_i
            for (int k = 0; k < Q; k++) {
                vx += f(i, j, k) * c[k][0];
                vy += f(i, j, k) * c[k][1];
            }

            // Normalize by density to get velocity
            if (rho(i, j) > 1e-10) {
                velocity(i, j, 0) = vx / rho(i, j);
                velocity(i, j, 1) = vy / rho(i, j);
            } else {
                velocity(i, j, 0) = 0.0;
                velocity(i, j, 1) = 0.0;
            }
        });

    Kokkos::fence();
}

void LBMStreaming::streaming() {
    // THE MAIN FOCUS OF THIS MILESTONE
    // Streaming operator with collision term = 0 (transport in vacuum)
    // f_i(r+c_i*Δt, t+Δt) = f_i(r,t)

    // Step 1: Stream particles to new positions
    Kokkos::parallel_for("streaming",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {NX, NY, Q}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            // Calculate source position (where this particle came from)
            // Apply periodic boundary conditions
            int src_x = (i - c[k][0] + NX) % NX;
            int src_y = (j - c[k][1] + NY) % NY;

            // Stream: f_new at destination gets f from source
            f_new(i, j, k) = f(src_x, src_y, k);
        });

    // Step 2: Copy streamed values back to original array
    Kokkos::parallel_for("copy_back",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {NX, NY, Q}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            f(i, j, k) = f_new(i, j, k);
        });

    Kokkos::fence();
}

void LBMStreaming::write_fields(int timestep) {
    // Copy data to host for file output
    auto h_rho = Kokkos::create_mirror_view(rho);
    auto h_velocity = Kokkos::create_mirror_view(velocity);

    Kokkos::deep_copy(h_rho, rho);
    Kokkos::deep_copy(h_velocity, velocity);

    // Write density field
    std::string rho_filename = "density_" + std::to_string(timestep) + ".dat";
    std::ofstream rho_file(rho_filename);
    rho_file << std::scientific << std::setprecision(6);

    for (int j = NY-1; j >= 0; j--) { // Top to bottom for visualization
        for (int i = 0; i < NX; i++) {
            rho_file << h_rho(i, j) << " ";
        }
        rho_file << "\n";
    }
    rho_file.close();

    // Write velocity field for matplotlib streamplot visualization
    std::string vel_filename = "velocity_" + std::to_string(timestep) + ".dat";
    std::ofstream vel_file(vel_filename);
    vel_file << std::scientific << std::setprecision(6);

    for (int j = NY-1; j >= 0; j--) { // Top to bottom for visualization
        for (int i = 0; i < NX; i++) {
            vel_file << h_velocity(i, j, 0) << " " << h_velocity(i, j, 1) << " ";
        }
        vel_file << "\n";
    }
    vel_file.close();

    std::cout << "Written fields for timestep " << timestep << std::endl;
}

void LBMStreaming::run_simulation(int num_timesteps) {
    std::cout << "=== LBM Milestone 2: Streaming Operator ===" << std::endl;
    std::cout << "Transport equation in vacuum (collision term = 0)" << std::endl;
    std::cout << "Grid size: " << NX << "x" << NY << " (as specified)" << std::endl;
    std::cout << "Using Kokkos Views as core data structure" << std::endl;
    std::cout << "Timesteps: " << num_timesteps << std::endl;

    for (int t = 0; t <= num_timesteps; t++) {
        // Compute macroscopic quantities from distribution function
        compute_density();
        compute_velocity();

        // Output fields every 2 timesteps for visualization
        if (t % 2 == 0) {
            write_fields(t);
        }

        // Perform streaming step (except on last timestep)
        if (t < num_timesteps) {
            streaming();
        }

        std::cout << "Completed timestep " << t << std::endl;
    }

    std::cout << "=== Streaming simulation completed! ===" << std::endl;
    std::cout << "Generated files for Python/matplotlib visualization" << std::endl;
}