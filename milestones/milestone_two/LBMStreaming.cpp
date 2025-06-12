#include "LBMStreaming.h"
#include <iostream>
#include <fstream>
#include <iomanip>

LBMStreaming::LBMStreaming() :
    f(NX, std::vector<std::vector<double>>(NY, std::vector<double>(Q, 0.0))),
    f_new(NX, std::vector<std::vector<double>>(NY, std::vector<double>(Q, 0.0))),
    rho(NX, std::vector<double>(NY, 0.0)),
    velocity(NX, std::vector<std::vector<double>>(NY, std::vector<double>(2, 0.0))) {

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
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            // Initialize all distribution functions to zero
            for (int k = 0; k < Q; k++) {
                f[i][j][k] = 0.0;
                f_new[i][j][k] = 0.0;
            }

            // Create a simple pattern - particles at center moving right
            if (i >= 6 && i <= 8 && j >= 4 && j <= 5) {
                f[i][j][1] = 1.0; // Direction 1 is right
                f[i][j][0] = 0.1; // Some stationary particles
            }

            // Add some particles moving up-right
            if (i >= 3 && i <= 5 && j >= 2 && j <= 3) {
                f[i][j][5] = 0.8; // Direction 5 is up-right
            }

            // Initialize macroscopic quantities
            rho[i][j] = 0.0;
            velocity[i][j][0] = 0.0;
            velocity[i][j][1] = 0.0;
        }
    }
}

void LBMStreaming::compute_density() {
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            double sum = 0.0;
            for (int k = 0; k < Q; k++) {
                sum += f[i][j][k];
            }
            rho[i][j] = sum;
        }
    }
}

void LBMStreaming::compute_velocity() {
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            double vx = 0.0, vy = 0.0;

            // Calculate momentum
            for (int k = 0; k < Q; k++) {
                vx += f[i][j][k] * c[k][0];
                vy += f[i][j][k] * c[k][1];
            }

            // Normalize by density to get velocity
            if (rho[i][j] > 1e-10) {
                velocity[i][j][0] = vx / rho[i][j];
                velocity[i][j][1] = vy / rho[i][j];
            } else {
                velocity[i][j][0] = 0.0;
                velocity[i][j][1] = 0.0;
            }
        }
    }
}

void LBMStreaming::streaming() {
    // Copy current distribution to new array with streaming
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < Q; k++) {
                // Calculate source position with periodic boundary conditions
                int src_x = (i - c[k][0] + NX) % NX;
                int src_y = (j - c[k][1] + NY) % NY;

                // Stream: f_new at current position gets f from source position
                f_new[i][j][k] = f[src_x][src_y][k];
            }
        }
    }

    // Copy new distribution back to original array
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < Q; k++) {
                f[i][j][k] = f_new[i][j][k];
            }
        }
    }
}

void LBMStreaming::write_fields(int timestep) {
    // Write density field
    std::string rho_filename = "density_" + std::to_string(timestep) + ".dat";
    std::ofstream rho_file(rho_filename);
    rho_file << std::scientific << std::setprecision(6);

    for (int j = NY-1; j >= 0; j--) { // Write from top to bottom for visualization
        for (int i = 0; i < NX; i++) {
            rho_file << rho[i][j] << " ";
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
            vel_file << velocity[i][j][0] << " " << velocity[i][j][1] << " ";
        }
        vel_file << "\n";
    }
    vel_file.close();

    std::cout << "Written fields for timestep " << timestep << std::endl;
}

void LBMStreaming::run_simulation(int num_timesteps) {
    std::cout << "Starting LBM Streaming Simulation (Standard C++ implementation)" << std::endl;
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