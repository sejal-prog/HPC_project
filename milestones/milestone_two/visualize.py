import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

def load_data(timestep):
    """Load density and velocity data for a given timestep"""
    try:
        # Load density
        rho_file = f"density_{timestep}.dat"
        rho = np.loadtxt(rho_file)

        # Load velocity
        vel_file = f"velocity_{timestep}.dat"
        vel_data = np.loadtxt(vel_file)

        # Reshape velocity data (each row has alternating vx, vy values)
        ny, nx2 = vel_data.shape
        nx = nx2 // 2
        vx = vel_data[:, ::2]  # Every other column starting from 0
        vy = vel_data[:, 1::2]  # Every other column starting from 1

        return rho, vx, vy
    except FileNotFoundError:
        return None, None, None

def plot_timestep(timestep):
    """Plot density and velocity field for a single timestep"""
    rho, vx, vy = load_data(timestep)

    if rho is None:
        print(f"Data for timestep {timestep} not found")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot density
    im1 = ax1.imshow(rho, cmap='viridis', origin='lower')
    ax1.set_title(f'Density (t={timestep})')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(im1, ax=ax1)

    # Plot velocity field
    y, x = np.mgrid[0:rho.shape[0], 0:rho.shape[1]]
    ax2.streamplot(x, y, vx.T, vy.T, density=1.5, color='blue', arrowsize=1.5)
    ax2.set_title(f'Velocity Field (t={timestep})')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_xlim(0, rho.shape[1]-1)
    ax2.set_ylim(0, rho.shape[0]-1)

    plt.tight_layout()
    plt.savefig(f'lbm_timestep_{timestep}.png', dpi=150, bbox_inches='tight')
    plt.show()

def create_animation():
    """Create animation of the streaming process"""
    timesteps = []
    data = []

    # Find available timesteps
    for t in range(0, 11, 2):  # Check timesteps 0, 2, 4, 6, 8, 10
        rho, vx, vy = load_data(t)
        if rho is not None:
            timesteps.append(t)
            data.append((rho, vx, vy))

    if not data:
        print("No data found for animation")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    def animate(frame):
        ax1.clear()
        ax2.clear()

        rho, vx, vy = data[frame]
        t = timesteps[frame]

        # Plot density
        im1 = ax1.imshow(rho, cmap='viridis', origin='lower')
        ax1.set_title(f'Density (t={t})')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')

        # Plot velocity field
        y, x = np.mgrid[0:rho.shape[0], 0:rho.shape[1]]
        ax2.streamplot(x, y, vx.T, vy.T, density=1.5, color='blue', arrowsize=1.5)
        ax2.set_title(f'Velocity Field (t={t})')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_xlim(0, rho.shape[1]-1)
        ax2.set_ylim(0, rho.shape[0]-1)

    anim = animation.FuncAnimation(fig, animate, frames=len(data), interval=800, repeat=True)
    plt.tight_layout()

    # Save animation
    anim.save('lbm_streaming.gif', writer='pillow', fps=1.2, dpi=100)
    plt.show()

if __name__ == "__main__":
    print("LBM Streaming Visualization")
    print("==========================")

    print("Available timesteps:")
    for t in range(0, 11, 2):
        if os.path.exists(f"density_{t}.dat"):
            print(f"  Timestep {t}")

    print("\nPlotting initial and final states...")
    # Plot initial and final states
    plot_timestep(0)   # Initial state
    plot_timestep(10)  # Final state

    print("\nCreating animation...")
    # Create animation
    create_animation()

    print("Visualization complete!")