#include "LBMStreaming.h"
#include <iostream>

/**
 * @brief Main function for Milestone 2 - LBM Streaming Operator
 *
 * This program demonstrates the streaming step of the Lattice Boltzmann Method
 * using the D2Q9 scheme. It simulates particle transport in vacuum (no collisions)
 * with periodic boundary conditions.
 *
 * Key features:
 * - D2Q9 velocity discretization (9 directions)
 * - Streaming operator: f_i(r+c_i*Δt, t+Δt) = f_i(r,t)
 * - Periodic boundary conditions
 * - 15x10 grid as specified
 * - File output for visualization
 */
int main(int argc, char* argv[]) {
    // Initialize Kokkos runtime
    Kokkos::initialize(argc, argv);
    {
        std::cout << "=== LBM Milestone 2: Streaming Operator ===" << std::endl;
        std::cout << "Kokkos initialized successfully" << std::endl;

        // Create LBM simulation object
        LBMStreaming lbm;

        // Run simulation for 10 timesteps
        lbm.run_simulation(10);

        std::cout << "=== Simulation completed successfully! ===" << std::endl;
        std::cout << "Generated files: density_*.dat and velocity_*.dat" << std::endl;
        std::cout << "Use 'python visualize.py' to create plots and animation" << std::endl;
    }
    // Finalize Kokkos runtime
    Kokkos::finalize();

    return 0;
}