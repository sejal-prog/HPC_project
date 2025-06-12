#include "LBMStreaming.h"
#include <iostream>

/**
 * @brief Main function for Milestone 2 - LBM Streaming Operator
 *
 * This program implements the streaming step of the Lattice Boltzmann Method
 * as specified in the assignment:
 *
 * - D2Q9 velocity discretization (9 directions)
 * - Streaming operator with collision term = 0 (transport in vacuum)
 * - Periodic boundary conditions
 * - 15x10 grid as specified
 * - Uses Kokkos Views as core data structure
 * - File output for Python/matplotlib visualization
 *
 * Focus: Understanding particle transport without interactions
 */
int main(int argc, char* argv[]) {
    // Initialize Kokkos runtime as required for Kokkos Views
    Kokkos::initialize(argc, argv);
    {
        std::cout << "=== LBM Milestone 2: Streaming Operator ===" << std::endl;
        std::cout << "Kokkos initialized successfully" << std::endl;
        std::cout << "Implementing transport equation in vacuum (no collisions)" << std::endl;

        // Create LBM streaming simulation object
        LBMStreaming lbm;

        // Run simulation for specified number of timesteps
        lbm.run_simulation(10);

        std::cout << "=== Milestone 2 completed successfully! ===" << std::endl;
        std::cout << "Use Python/matplotlib to visualize the results" << std::endl;
        std::cout << "Suggestion: use streamplot function for velocity field visualization" << std::endl;
    }
    // Finalize Kokkos runtime
    Kokkos::finalize();

    return 0;
}