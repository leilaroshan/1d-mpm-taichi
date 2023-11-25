import taichi as ti
import math
import numpy as np
import matplotlib.pyplot as plt

ti.init(arch=ti.cpu)

# Computational grid
L = 1                        # domain size
nodes      = ti.Vector.field(2, dtype=ti.f32, shape=L)
nodes[0]   = [0, 1]
nnodes     =2                # len (nodes)=2
nelements  = 1               # number of elements
nparticles = 1               # number of particles
el_length  = L / nelements   # element length

# Initial conditions
v0    = 0.1                 # initial velocity

# Material property
E   = 4 * (np.pi)**2        # Young's modulus
rho = 1.0                   # Density

# Material points as scalar value
x_p      = ti.field(ti.f32, shape=())
mass_p   = ti.field(dtype=ti.f32, shape=())
vol_p    = ti.field(dtype=ti.f32, shape=())
vel_p    = ti.field(dtype=ti.f32, shape=())
stress_p = ti.field(dtype=ti.f32, shape=())
strain_p = ti.field(dtype=ti.f32, shape=())
mv_p = ti.field(dtype=ti.f32, shape=())

# Time
duration = 10
dt       = 0.01
time     = 0
nsteps   = int(duration / dt)    # scalar value for nstep


# Store time, velocity, and position with time
time_t = ti.field(ti.f32, shape=nsteps)
vel_t  = ti.field(ti.f32, shape=nsteps)
x_t    = ti.field(ti.f32, shape=nsteps)
se_t   = ti.field(ti.f32, shape=nsteps)
ke_t   = ti.field(ti.f32, shape=nsteps)
te_t   = ti.field(ti.f32, shape=nsteps)
strain_rate_p = ti.field(ti.f32, shape=())
dstrain_p     = ti.field(ti.f32, shape=())

# shape function and derivative
N  = ti.Vector.field(2, dtype=ti.f32 , shape=())
dN = ti.Vector.field(2, dtype=ti.f32, shape=())

# particle mass, momentum and velocity of nodes
mass_n = ti.Vector.field(2, dtype=ti.f32, shape=())
mv_n   = ti.Vector.field(2, dtype=ti.f32, shape=())
vel_n  = ti.Vector.field(2, dtype=ti.f32, shape=())

# compute nodal force
f_ext_n   = ti.Vector.field(2, dtype=ti.f32, shape=())
f_int_n   = ti.Vector.field(2, dtype=ti.f32, shape=())
f_total_n = ti.Vector.field(2, dtype=ti.f32, shape=())

# Initialize Taichi tensors
@ti.func
def initialize():
    x_p[None]      = 0.5 * el_length
    mass_p[None]   = rho*L 
    vol_p[None]    = el_length / nparticles
    vel_p[None]    = v0
    stress_p[None] = 0.0
    strain_p[None] = 0.0
    mv_p[None]     = mass_p[None] * vel_p[None]
    f_ext_n[None]  = [0, 0] 

@ti.func
def compute_N():   # range = nsteps
    # shape function and derivative
    N[None][0]  = 1 - abs(x_p[None] - nodes[0][0]) / L
    N[None][1]  = 1 - abs(x_p[None] - nodes[0][1]) / L
    dN[None][0] = -1 / L
    dN[None][1] = 1 / L

@ti.func
def compute_mass_and_momentum():
    # map particle mass and momentum to nodes 
    mass_n[None] = N[None] * mass_p[None]   # Mapped mass at the nodes: [0.5 0.5]
    mv_n[None] = N[None] * mv_p[None]       # Momentum at nodes: [0.05 0.05]
    # apply boundary condition: velocity at left node is zero
    mv_n[None][0] = 0.0   


@ti.func
def internal_and_external_force():
    f_int_n[None] = - dN[None] * vol_p[None] * stress_p[None]
    f_ext_n[None] = [0, 0]


#Calculate the total unbalanced nodal forces
@ti.func
def total_force():
    f_total_n[None] = f_int_n[None] + f_ext_n[None]
    # apply boundary condition: left node has no acceleration (f = m * a, and a = 0)
    f_total_n[None][0] = 0.0
    # update nodal momentum
    mv_n[None] += f_total_n[None] * dt
####################################################################
# Updating 
@ti.func
def update_particle():
    # update particle position and velocity 
    for i in range(nnodes):
         vel_p[None] += dt * N[None][i] * f_total_n[None][i] / mass_n[None][i]
         x_p[None]   += dt * N[None][i] * mv_n[None][i] / mass_n[None][i]
    # update particle momentum
    mv_p[None] = mass_p[None] * vel_p[None]
    # update map nodal velocity
    vel_n[None] = mass_p[None] * vel_p[None] * (N[None]/ mass_n[None])
    # Apply boundary condition and set left nodal velocity to zero
    vel_n[None][0] = 0
    
    # Compute strains and stresses
    strain_rate_p[None] = dN[None][0] * vel_n[None][0] + dN[None][1] * vel_n[None][1]
    # compute strain increament 
    dstrain_p[None]  = strain_rate_p[None] * dt
    strain_p[None]  += dstrain_p[None]
    # compute stress
    stress_p[None] += E * dstrain_p[None]

@ti.kernel
def mpm():
    initialize()
    ti.loop_config(serialize=True)
    for k in range(nsteps):
        compute_N()
        compute_mass_and_momentum() 
        internal_and_external_force()
        total_force()
        update_particle()
            
        time_t[k] = dt * k
        vel_t[k]  = vel_p[None]
        x_t[k]    = x_p[None]

mpm()
print(vel_p)

plt.plot(time_t, vel_t, 'ob', markersize = 2, label='mpm')
plt.xlabel('time (s)')
plt.ylabel('velocity (m/s)')
plt.legend()
#plt.show()
plt.savefig('tv.png')
