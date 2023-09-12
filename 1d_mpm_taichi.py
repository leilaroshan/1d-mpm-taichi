import taichi as ti
import math
import numpy as np
import matplotlib.pyplot as plt

ti.init(arch=ti.cpu)

# Computational grid
L = 1                        # domain size
nodes      = ti.Vector.field(2, dtype=ti.f32, shape=L)
nodes[0]   = [0, 1]
nnodes=2                     # len (nodes)=2
nelements  = 1               # number of elements
nparticles = 1               # number of particles
el_length  = L / nelements   # element length


# Initial conditions
v0    = 0.1                 # initial velocity
x_loc = 0.5                 # location to get analytical solution

# Material property
E   = 4 * (np.pi)**2        # Young's modulus
rho = 1.0                   # Density

# Material points as scalar value
x_p = ti.field(ti.f32, shape=())
mass_p = ti.field(dtype=ti.f32, shape=())
vol_p = ti.field(dtype=ti.f32, shape=())
vel_p = ti.field(dtype=ti.f32, shape=())
stress_p = ti.field(dtype=ti.f32, shape=())
strain_p = ti.field(dtype=ti.f32, shape=())
mv_p = ti.field(dtype=ti.f32, shape=())

# Time
duration = 10
dt       = 0.01
time     = 0
# nsteps   = int(duration/dt)
nsteps  = int(duration / dt)    # scalar value for nstep

# Store time, velocity, and position with time
time_t = ti.field(ti.f32, shape=nsteps)
vel_t = ti.field(ti.f32, shape=nsteps)
x_t = ti.field(ti.f32, shape=nsteps)
se_t = ti.field(ti.f32, shape=nsteps)
ke_t = ti.field(ti.f32, shape=nsteps)
te_t = ti.field(ti.f32, shape=nsteps)

# shape function and derivative
N = ti.Vector.field(2, dtype=ti.f32 , shape=())
dN = ti.Vector.field(2, dtype=ti.f32, shape=())

mass_n = ti.field(dtype=ti.f32, shape=nnodes)
mv_n = ti.field(dtype=ti.f32, shape=nnodes)

# Initialize Taichi tensors
@ti.kernel
def initialize():
    x_p[None] = 0.5 * el_length
    mass_p[None] = 1.0
    vol_p[None] = el_length / nparticles
    vel_p[None] = v0
    stress_p[None] = 0.0
    strain_p[None] = 0.0
    mv_p[None] = mass_p[None] * vel_p[None]

# Time-stepping loop
for _ in range(nsteps):
    initialize()

@ti.kernel
def compute_N():   #range = nsteps
    for i in range(2):
        # shape function and derivative
        N[None][0]  = 1 - abs(x_p[None] - nodes[0][0]) / L
        N[None][1]  = 1 - abs(x_p[None] - nodes[0][1]) / L
        dN[None][0] = -1 / L
        dN[None][1] = 1 / L
        print(N[None])
        # map particle mass and momentum to nodes
        mass_n[None][0] = N[0] * mass_p[None]
        # mv_n[0]   = N[None] * mv_p[0]

compute_N()
