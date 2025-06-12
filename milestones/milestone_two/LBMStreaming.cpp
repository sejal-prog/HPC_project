//
// Created by sejal on 6/12/25.
//
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
    // D2Q9 velocity vectors
    c[0][0] = 0;  c[0][1] = 0;   // stationary
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
    // Initialize with simple test pattern - particles moving right
    Kokkos::parallel_for("initialize",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {NX, NY}),
        KOKKOS_LAMBDA(const int i, const int j) {
            // Initialize all distribution functions to zero
            for (int k = 0; k < Q; k++) {
                f(i, j, k) = 0.0;
                f_new(i, j, k) = 0.0;
            }

            // Create a simple pattern - particles at center moving right
            if (i >= 6 && i <= 8 && j >= 4 && j <= 5) {
                f(i, j, 1) = 1.0; // Direction 1 is right
                f(i, j, 0) = 0.1; // Some stationary particles
            }

            // Add some particles moving up-right
            if (i >= 3 && i <= 5 && j >= 2 && j <= 3) {
                f(i, j, 5) = 0.8; // Direction 5 is up-right
            }

            // Initialize macroscopic quantities
            rho(i, j) = 0.0;
            velocity(i, j, 0) = 0.0;
            velocity(i, j, 1) = 0.0;
        });

    Kokkos::fence();
}

void LBMStreaming::compute_density() {
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
    Kokkos::parallel_for("compute_velocity",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {NX, NY}),
        KOKKOS_LAMBDA(const int i, const int j) {
            double vx = 0.0, vy = 0.0;

            // Calculate momentum
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
    // Copy current distribution to new array with streaming
    Kokkos::parallel_for("streaming",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {NX, NY, Q}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            // Calculate source position with periodic boundary conditions
            // We need to find where the particle at (i,j) came from
            int src_x = (i - c[k][0] + NX) % NX;
            int src_y = (j - c[k][1] + NY) % NY;

            // Stream: f_new at current position gets f from source position
            f_new(i, j, k) = f(src_x, src_y, k);
        });

    // Copy new distribution back to original array
    Kokkos::parallel_for("copy_back",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {NX, NY, Q}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            f(i, j, k) = f_new(i, j, k);
        });

    Kokkos::fence();
}

void LBMStreaming::write_fields(int timestep) {
    // Copy data to host for output
    auto h_rho = Kokkos::create_mirror_view(rho);
    auto h_velocity = Kokkos::create_mirror_view(velocity);

    Kokkos::deep_copy(h_rho, rho);
    Kokkos::deep_copy(h_velocity, velocity);

    // Write density field
    std::string rho_filename = "density_" + std::to_string(timestep) + ".dat";
    std::ofstream rho_file(rho_filename);
    rho_file << std::scientific << std::setprecision(6);

    for (int j = NY-1; j >= 0; j--) { // Write from top to bottom for visualization
        for (int i = 0; i < NX; i++) {
            rho_file << h_rho(i, j) << " ";
        }
        rho_file << "\n";
    }
    rho_file.close();

    // Write velocity field
    std::string vel_filename = "velocity_" + std::to_string(timestep) + ".dat";
    std::ofstream vel_file(vel_filename);
    vel_file << std::scientific << std::setprecision(6);

    for (int j = NY-1; j >= 0; j--) { // Write from top to bottom for visualization
        for (int i = 0; i < NX; i++) {
            vel_file << h_velocity(i, j, 0) << " " << h_velocity(i, j, 1) << " ";
        }
        vel_file << "\n";
    }
    vel_file.close();

    std::cout << "Written fields for timestep " << timestep << std::endl;
}

void LBMStreaming::run_simulation(int num_timesteps) {
    std::cout << "Starting LBM Streaming Simulation" << std::endl;
    std::cout << "Grid size: " << NX << "x" << NY << std::endl;
    std::cout << "Timesteps: " << num_timesteps << std::endl;

    for (int t = 0; t <= num_timesteps; t++) {
        // Compute macroscopic quantities
        compute_density();
        compute_velocity();

        // Output fields every 2 timesteps
        if (t % 2 == 0) {
            write_fields(t);
        }

        // Perform streaming step (except on last timestep)
        if (t < num_timesteps) {
            streaming();
        }

        std::cout << "Completed timestep " << t << std::endl;
    }
}