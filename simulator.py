import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import cmasher as cmr
from tqdm import tqdm
import matplotlib.animation as animation
import numpy as np

# Standardization
N_ITERATIONS = 70000
RENOLDS_NUMBER = 80
N_POINTS_X=300
N_POINTS_Y=50


# cylinders positions 
CYLINDER_CENTER_INDEX_X = N_POINTS_X//5
CYLINDER_CENTER_INDEX_Y = N_POINTS_Y //2
CYLINDER_CENTER_INDICES = N_POINTS_Y //9

MAX_HORIZONTAL_INFLOW_VELOCITY = 0.04

VISUALIZE= True
PLOT_EVERY_N_STEPS=10
SKIP_FIRST_N_ITERATIONS = 0

N_DISCRETE_VELOCITIES = 9

LATTICE_VELOCITIES = jnp.array([
    [0, 1, 0, -1, 0, 1, -1, -1, 1],
    [0, 0, 1, 0, -1, 1, 1, -1, -1]
])

LATTICE_INDICIES = jnp.array([
    0,1,2,3,4,5,6,7,8
])

OPPOSITE_LATTICE_INDICES = jnp.array([
    0,3,4,1,2,7,8,5,6
])

#used in equilibrium computation
LATTICE_WEIGHTS = jnp.array([
    4/9, # center velocity (0)
    1/9, 1/9, 1/9, 1/9, #axis Aligned velocities
    1/36, 1/36, 1/36, 1/36 # diagonal velocities
])

RIGHT_VELOCITIES = jnp.array([1,5,8])
UP_VELOCITIES = jnp.array([2,5,6])
LEFT_VELOCITIES = jnp.array([3,6,7])
DOWN_VELOCITIES = jnp.array([4,7,8])
PURE_VERTICLE_VELOCITIES = jnp.array([0,2,4])
PURE_HORIZONTAL_VELOCITIES = jnp.array([0,1,3])

# Creating Helper Functions

def get_density(discrete_velocities):
    density = jnp.sum(discrete_velocities,axis=-1)
    return density

def get_macroscopic_velocities(discrete_velocities, density):
    density = jnp.maximum(density, 1e-12)
    macroscopic_velocities = jnp.einsum(
        "NMQ,dQ->NMd",
        discrete_velocities,
        LATTICE_VELOCITIES,
    ) / density[..., jnp.newaxis]
    return macroscopic_velocities

    
def get_equilibrium_discrete_velocities(macroscopic_velocities, density):
    # Project macroscopic velocity onto discrete lattice directions
    projected_discrete_velocities = jnp.tensordot(
    macroscopic_velocities,  # (N, M, d)
    LATTICE_VELOCITIES,      # (d, Q)
    axes=([2], [0])          # sum over d
)

    # Compute |u|^2 over last axis
    macroscopic_velocity_magnitude = jnp.sum(macroscopic_velocities**2, axis=-1)

    # Compute equilibrium distribution
    equilibrium_discrete_velocities = (
        density[..., jnp.newaxis]
        * LATTICE_WEIGHTS[jnp.newaxis, jnp.newaxis, :]
        * (
            1
            + 3 * projected_discrete_velocities
            + 9/2 * projected_discrete_velocities**2
            - 3/2 * macroscopic_velocity_magnitude[..., jnp.newaxis]
        )
    )

    return equilibrium_discrete_velocities


def main():
    jax.config.update("jax_enable_x64",True)
    kinematic_viscosity = ( MAX_HORIZONTAL_INFLOW_VELOCITY * CYLINDER_CENTER_INDICES) / (RENOLDS_NUMBER)
    relaxation_omega = 1.0 / (3.0 * kinematic_viscosity + 0.5)
    
    
    x = jnp.arange(N_POINTS_X)
    y = jnp.arange(N_POINTS_Y)

    X, Y = jnp.meshgrid(x, y, indexing="ij")

    
    #obsticle mask: an array of hte same shape like x,y but contains true if hte object belongs to object
    obstacle_mask= jnp.sqrt( (X-CYLINDER_CENTER_INDEX_X)**2 + ( Y - CYLINDER_CENTER_INDEX_Y)**2 ) <CYLINDER_CENTER_INDICES
    
    velocity_profile = jnp.zeros((N_POINTS_X,N_POINTS_Y,2))
    velocity_profile = velocity_profile.at[:, :, 0].set(MAX_HORIZONTAL_INFLOW_VELOCITY)
    
    @jax.jit 
    def update(discrete_velocities_prev):
        discrete_velocities_prev = discrete_velocities_prev.at[-1,:,LEFT_VELOCITIES].set(discrete_velocities_prev[-2,:,LEFT_VELOCITIES])
        density_prev = get_density(discrete_velocities_prev)
        macroscopic_velocities_prev = get_macroscopic_velocities(discrete_velocities_prev,density_prev)
        macroscopic_velocities_prev = macroscopic_velocities_prev.at[0,1:-1,:].set(velocity_profile[0,1:-1,:])
        density_prev = density_prev.at[0,:].set(
            get_density(discrete_velocities_prev[0,:,PURE_VERTICLE_VELOCITIES].T) +
            2 *
            get_density(discrete_velocities_prev[0,:,LEFT_VELOCITIES].T)
            / (1 - macroscopic_velocities_prev[0,:, 0])
        )

        equilibrium_discrete_velocities = get_equilibrium_discrete_velocities(
            macroscopic_velocities_prev,
            density_prev
        )
        discrete_velocities_prev= discrete_velocities_prev.at[0,:,RIGHT_VELOCITIES].set(
            equilibrium_discrete_velocities[0,:,RIGHT_VELOCITIES]
        )
        
        discrete_velocities_post_collision = (
            discrete_velocities_prev - relaxation_omega * (discrete_velocities_prev - equilibrium_discrete_velocities)
        )
        
        for i in range(N_DISCRETE_VELOCITIES):
            discrete_velocities_post_collision= discrete_velocities_post_collision.at[obstacle_mask,LATTICE_INDICIES[i]].set(
                    discrete_velocities_prev[obstacle_mask,OPPOSITE_LATTICE_INDICES[i]]
                )

        discrete_velocities_streamed = discrete_velocities_post_collision
        for i in range(N_DISCRETE_VELOCITIES):
            discrete_velocities_streamed = discrete_velocities_streamed.at[:,:,i].set(
                jnp.roll(
                    jnp.roll(
                        discrete_velocities_post_collision[:,:,i],
                        LATTICE_VELOCITIES[0,i],
                        axis=0),
                    LATTICE_VELOCITIES[1,i],
                    axis=1,
                    )
                )
        return discrete_velocities_streamed
    
    discrete_velocities_prev = get_equilibrium_discrete_velocities(
        velocity_profile,
        jnp.ones((N_POINTS_X,N_POINTS_Y))
    )
    
    # plt.style.use("dark_background")
    # plt.figure(figsize=(15,6),dpi=100)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    
    # Setup FFMpeg writer
    writer = animation.FFMpegWriter(fps=60)  # adjust fps as needed
    
    with writer.saving(fig, "longsimulation.mp4", dpi=150):
        for iteration_index in tqdm(range(N_ITERATIONS)):
            discrete_velocities_next = update(discrete_velocities_prev)
            discrete_velocities_prev = discrete_velocities_next

            if iteration_index % PLOT_EVERY_N_STEPS == 0 and iteration_index > SKIP_FIRST_N_ITERATIONS:
                density = get_density(discrete_velocities_next)
                macroscopic_velocities = get_macroscopic_velocities(
                    discrete_velocities_next,
                    density
                )
                velocity_magnitude = jnp.linalg.norm(macroscopic_velocities, axis=-1, ord=2)

                du_dx, du_dy = jnp.gradient(macroscopic_velocities[..., 0])
                dv_dx, dv_dy = jnp.gradient(macroscopic_velocities[..., 1])
                curl = du_dy - dv_dx

                ax1.clear()
                ax1.contourf(X, Y, velocity_magnitude, levels=30, cmap=cmr.amber)
                ax1.add_patch(plt.Circle(
                    (CYLINDER_CENTER_INDEX_X, CYLINDER_CENTER_INDEX_Y),
                    CYLINDER_CENTER_INDICES,
                    color="darkgreen"
                ))
                ax1.set_title("Velocity Magnitude")

                ax2.clear()
                ax2.contourf(X, Y, curl, levels=50, cmap=cmr.redshift, vmin=-0.02, vmax=0.02)
                ax2.add_patch(plt.Circle(
                    (CYLINDER_CENTER_INDEX_X, CYLINDER_CENTER_INDEX_Y),
                    CYLINDER_CENTER_INDICES,
                    color="darkgreen"
                ))
                ax2.set_title("Curl")

                # write this frame to the video
                writer.grab_frame()
    if VISUALIZE:
        plt.show()

if __name__=="__main__":
    main()
