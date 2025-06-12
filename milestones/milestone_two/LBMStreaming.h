#ifndef LBM_STREAMING_H
#define LBM_STREAMING_H

#include <Kokkos_Core.hpp>

/**
 * @class LBMStreaming
 * @brief Lattice Boltzmann Method streaming operator implementation
 *
 * This class implements the streaming step of the LBM for D2Q9 scheme.
 * It handles particle transport in vacuum (zero collision term) with
 * periodic boundary conditions.
 *
 * Uses Kokkos Views as core data structure as specified in the assignment.
 */
class LBMStreaming {
private:
    static constexpr int Q = 9;   ///< Number of directions in D2Q9 scheme
    static constexpr int NX = 15; ///< Grid width (as specified)
    static constexpr int NY = 10; ///< Grid height (as specified)

    /// D2Q9 velocity vectors [direction][x/y component]
    int c[Q][2];

    /// Distribution function [x][y][direction] - "number of particles"
    /// Using Kokkos View as specified in the assignment
    Kokkos::View<double***> f;

    /// New distribution function after streaming [x][y][direction]
    Kokkos::View<double***> f_new;

    /// Density field [x][y] - computed from f
    Kokkos::View<double**> rho;

    /// Velocity field [x][y][component(x/y)] - computed from f
    Kokkos::View<double***> velocity;

public:
    /**
     * @brief Constructor - initializes Kokkos views and velocity vectors
     */
    LBMStreaming();

    /**
     * @brief Initialize the distribution function with simple test patterns
     *
     * As suggested: "start with values only in few (or even one) direction"
     * Think of f_i as "number of particles" at position (x,y) moving in direction i
     */
    void initialize();

    /**
     * @brief Compute density at each lattice point
     *
     * Calculates ρ(x,y) = Σ f_i(x,y) over all directions i
     * Uses Kokkos::parallel_for as specified
     */
    void compute_density();

    /**
     * @brief Compute velocity field at each lattice point
     *
     * Calculates u(x,y) = (1/ρ) * Σ f_i(x,y) * c_i
     * Uses Kokkos::parallel_for as specified
     */
    void compute_velocity();

    /**
     * @brief Perform streaming step - the main focus of this milestone
     *
     * Implements the streaming operator with collision term = 0:
     * f_i(r+c_i*Δt, t+Δt) = f_i(r,t)
     *
     * Shifts components of f along the grid according to their direction.
     * Uses Kokkos::parallel_for as specified in the assignment.
     * Applies periodic boundary conditions.
     */
    void streaming();

    /**
     * @brief Write density and velocity fields to files for visualization
     *
     * @param timestep Current simulation timestep for filename
     *
     * Outputs data for Python/matplotlib visualization as suggested
     */
    void write_fields(int timestep);

    /**
     * @brief Run streaming simulation (transport equation in vacuum)
     *
     * @param num_timesteps Number of timesteps to simulate
     *
     * Main simulation loop focusing on streaming operator:
     * - No collision (collision term = 0)
     * - Particle transport in vacuum
     * - Periodic boundary conditions
     */
    void run_simulation(int num_timesteps);

private:
    /**
     * @brief Initialize D2Q9 velocity vectors
     *
     * Sets up the 9 velocity directions for the D2Q9 scheme
     * as shown in Figure 1a of the specification
     */
    void setup_velocity_vectors();
};