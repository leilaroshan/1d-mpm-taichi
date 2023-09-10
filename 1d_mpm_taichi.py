import taichi as ti
import math
import numpy as np
import matplotlib.pyplot as plt

ti.init(arch=ti.cpu)

# Computational grid
L = 1                        # domain size
nodes      = ti.Vector.field(2, dtype=ti.f32, shape=L)
nodes[0]   = [0, 1]
nelements  = 1               # number of elements
nparticles = 1               # number of particles
el_length  = L / nelements   # element length


# Initial conditions
v0    = 0.1                 # initial velocity
x_loc = 0.5                 # location to get analytical solution

# Material property
E   = 4 * (np.pi)**2        # Young's modulus
rho = 1.0                   # Density

# Material points
x_p        = 0.5 * el_length         # position
# mass_p     = 1.0                     # mass
# vol_p      = el_length / nparticles  # volume
# vel_p      = v0                      # velocity
# stress_p   = 0.0                     # stress
# strain_p   = 0.0                     # strain
# mv_p       = mass_p * vel_p          # momentum = m * v

# Material points as scalar value
#x_p = ti.field(ti.f32, shape=())
mass_p = ti.field(dtype=ti.f32, shape=nparticles)
vol_p = ti.field(dtype=ti.f32, shape=nparticles)
vel_p = ti.field(dtype=ti.f32, shape=nparticles)
stress_p = ti.field(dtype=ti.f32, shape=nparticles)
strain_p = ti.field(dtype=ti.f32, shape=nparticles)
mv_p = ti.field(dtype=ti.f32, shape=nparticles)

# Time
duration = 10
dt       = 0.01
time     = 0
# nsteps   = int(duration/dt)
nsteps  = ti.field(dtype=ti.i32, shape=())    # scalar value for nstep
nsteps[None] = int(duration / dt)


# Store time, velocity, and position with time
#time_t, vel_t, x_t, se_t, ke_t, te_t = [], [], [], [], [], []

time_t = ti.field(ti.f32, shape=nelements)
vel_t = ti.field(ti.f32, shape=())
x_t = ti.field(ti.f32, shape=())
se_t = ti.field(ti.f32, shape=())
ke_t = ti.field(ti.f32, shape=())
te_t = ti.field(ti.f32, shape=())

# shape function and derivative
N = ti.Vector.field(2, dtype=ti.f32 , shape=())
dN = ti.Vector.field(2, dtype=ti.f32, shape=())

# mass_n = ti.field(dtype=ti.f32, shape=nnodes)
# mv_n = ti.field(dtype=ti.f32, shape=nnodes)

@ti.kernel
def compute_N():
    for _ in range(nsteps):
        # shape function and derivative
        N[0]  = 1 - abs(x_p[None] - nodes[0][0]) / L
        N[1]  = 1 - abs(x_p[None] - nodes[0][1]) / L
        dN[0] = -1 / L
        dN[1] = 1 / L
        print (N)
        # map particle mass and momentum to nodes
        mass_n = N * mass_p
        mv_n   = N * mv_p









