### 🌀 Lattice Boltzmann Fluid Simulation (JAX)

A fully parallelized fluid dynamics simulation based on the Lattice Boltzmann Method (LBM).
This project models laminar flow and demonstrates how flow behavior changes when disturbed by a centrally positioned obstacle.

The simulation is accelerated using JAX, enabling high-performance computation on both CPU and GPU.


<video src="longsimulation.mp4" controls width="600"></video>


##📌 Features

🧮 Lattice Boltzmann Method (LBM) implementation
⚡ Fully parallelized computation using JAX
🎥 Real-time visualization with Matplotlib
🎨 Scientific colormaps via CMasher
🔄 Animation support
📊 Progress tracking with tqdm
📂 Dependencies

This project requires:

jax
jaxlib
numpy
matplotlib
cmasher
tqdm

You can install general dependencies using:

pip install -r requirements.txt

##⚠️ Important: JAX Installation (CPU vs GPU)

JAX must be installed correctly depending on your hardware.

#🖥 CPU Installation (Recommended Default)

If you are running on CPU only:

pip install --upgrade pip
pip install jax jaxlib


Or directly via requirements:

pip install -r requirements.txt

#🚀 GPU Installation (CUDA Required)

If you have an NVIDIA GPU and CUDA installed, you must install the CUDA-specific JAX build.

For CUDA 13:

pip install --upgrade pip
pip install -U "jax[cuda13]"

After installation, verify GPU detection:

import jax
print(jax.devices())


If GPU is available, it will list a CUDA device.

##▶️ Running the Simulation

After installing dependencies:

python your_script_name.py


The simulation will:

Compute fluid flow using LBM
Visualize velocity fields
Generate animated results (if enabled)

##🧠 About the Method

The Lattice Boltzmann Method (LBM) is a mesoscopic numerical approach for simulating fluid dynamics.
Instead of directly solving the Navier–Stokes equations, LBM evolves particle distribution functions over a discrete lattice grid.

In this project:

Laminar inlet flow is initialized
A centered obstacle disturbs the flow
Velocity field evolution is visualized
Flow separation and wake formation can be observed

##📈 Performance

Because the simulation uses JAX, it benefits from:

Just-In-Time (JIT) compilation
Automatic vectorization
GPU acceleration (if available)
Highly parallel array operations

This makes it significantly faster than naive NumPy-based implementations.
