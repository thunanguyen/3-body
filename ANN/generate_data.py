import numpy as np
import json
import matplotlib.pyplot as plt

"""
Create Your Own N-body Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz
Simulate orbits of stars interacting due to gravity
Code calculates pairwise forces according to Newton's Law of Gravity
"""

def get_acc( pos, mass, G, softening ):
	"""
    Calculate the acceleration on each particle due to Newton's Law 
	pos  is an N x 3 matrix of positions
	mass is an N x 1 vector of masses
	G is Newton's Gravitational constant
	softening is the softening length
	a is N x 3 matrix of accelerations
	"""
	# positions r = [x,y,z] for all particles
	x = pos[:,0]
	y = pos[:,1]
	z = pos[:,2]

	# matrix that stores all pairwise particle separations: r_j - r_i
	dx = x.T - x
	dy = y.T - y
	dz = z.T - z

	# matrix that stores 1/r^3 for all particle pairwise particle separations 
	inv_r3 = (dx**2 + dy**2 + dz**2 + softening**2)
	inv_r3[inv_r3>0] = inv_r3[inv_r3>0]**(-1.5)

	ax = G * (dx * inv_r3) @ mass
	ay = G * (dy * inv_r3) @ mass
	az = G * (dz * inv_r3) @ mass
	
	# pack together the acceleration components
	a = np.hstack((ax,ay,az))

	return a

def n_body():
	""" N-body simulation """
	# Generate Initial Conditions
	#np.random.seed(17)            # set the random number generator seed
	
	# Simulation parameters
	N         = 3    # Number of particles
	t         = 0.0      # current time of the simulation
	tEnd      = np.random.uniform(0.0, 10.0) #10.0   # time at which simulation ends
	dt        = 0.0005   # timestep
	softening = 0.1    # softening length
	G         = 1.0    # Newton's Gravitational Constant
	plotRealTime = True # switch on for plotting as the simulation goes along
	
	mass = 20.0*np.ones((N,1))/N  # total mass of particles is 20
	init_pos  = np.random.randn(N,3)   # randomly selected positions and velocities
	init_vel  = np.random.randn(N,3)

	pos = init_pos
	vel = init_vel
	# Convert to Center-of-Mass frame
	vel -= np.mean(mass * vel,0) / np.mean(mass)
	
	# calculate initial gravitational accelerations
	acc = get_acc( pos, mass, G, softening )
	
	# number of timesteps
	Nt = int(np.ceil(tEnd/dt))
	
	# Simulation Main Loop
	for i in range(Nt):
		# (1/2) kick
		vel += acc * dt/2.0
		
		# drift
		pos += vel * dt
		
		# update accelerations
		acc = get_acc( pos, mass, G, softening )
		
		# (1/2) kick
		vel += acc * dt/2.0

	dt = tEnd - Nt
	vel += acc * dt/2.0
	pos += vel * dt
	acc = get_acc( pos, mass, G, softening )
	vel += acc * dt/2.0
	
	return init_pos, init_vel, tEnd, pos, vel

if __name__== "__main__":
        np.random.seed(17)
        
        for i in range(1000):
                with open(f'data/init_{i}.npy', 'wb') as f:
                        np.savez(f, pos=n_body()[0], vel=n_body()[1],  t=n_body()[2])
                        #init = n_body()[0].flatten() + n_body()[1].flatten()
                        #np.save(f, init)

                with open(f'data/final_{i}.npy', 'wb') as f:
                        np.savez(f, pos=n_body()[3], vel=n_body()[4])
                        #final = n_body()[2].flatten() + n_body()[3].flatten() + np.array([n_body()[4]])
                        #np.save(f, final)
