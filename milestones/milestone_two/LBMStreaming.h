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
 */
class LBMStreaming {
private:
    static constexpr int Q = 9;   ///< Number of directions in D2Q9 scheme
    static constexpr int NX = 15; ///< Grid width
    static constexpr int NY = 10; ///< Grid height

    /// D2Q9 velocity vectors [direction][x/y component]
    int c[Q][2];

    /// Distribution function [x][y][direction]
    Kokkos::View<double***> f;

    /// New distribution function after streaming [x][y][direction]
    Kokkos::View<double***> f_new;

    /// Density field [x][y]
    Kokkos::View<double**> rho;

    /// Velocity field [x][y][component(x/y)]
    Kokkos::View<double***> velocity;

public:
    /**
     * @brief Constructor - initializes Kokkos views and velocity vectors
     */
    LBMStreaming();

    /**
     * @brief Initialize the distribution function with test patterns
     *
     * Creates simple particle distributions moving in specific directions
     * for testing the streaming operator.
     */
    void initialize();

    /**
     * @brief Compute density at each lattice point
     *
     * Calculates ρ(x,y) = Σ f_i(x,y) over all directions i
     */
    void compute_density();

    /**
     * @brief Compute velocity field at each lattice point
     *
     * Calculates v(x,y) = (1/ρ) * Σ f_i(x,y) * c_i
     */
    void compute_velocity();

    /**
     * @brief Perform streaming step
     *
     * Implements f_i(r+c_i*Δt, t+Δt) = f_i(r,t) with periodic boundaries.
     * Uses Kokkos::parallel_for for performance.
     */
    void streaming();

    /**
     * @brief Write density and velocity fields to files
     *
     * @param timestep Current simulation timestep for filename
     *
     * Outputs data in format suitable for Python/matplotlib visualization
     */
    void write_fields(int timestep);

    /**
     * @brief Run complete streaming simulation
     *
     * @param num_timesteps Number of timesteps to simulate
     *
     * Executes the main simulation loop with density/velocity computation
     * and periodic file output.
     */
    void run_simulation(int num_timesteps);

private:
    /**
     * @brief Initialize D2Q9 velocity vectors
     *
     * Sets up the 9 velocity directions for the D2Q9 scheme
     */
    void setup_velocity_vectors();
};

#endif // LBM_STREAMING_H