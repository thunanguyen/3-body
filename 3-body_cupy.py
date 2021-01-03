import time
import argparse
import cupy as cp
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
	x = pos[:,0:1]
	y = pos[:,1:2]
	z = pos[:,2:3]

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
	a = cp.hstack((ax,ay,az))

	return a

def main(tEnd):
	""" N-body simulation """
	
	# Simulation parameters
	N         = 3    # Number of particles
	t         = 0      # current time of the simulation
	tEnd      = tEnd   # time at which simulation ends
	dt        = 0.0005   # timestep
	softening = 0.1    # softening length
	G         = 1.0    # Newton's Gravitational Constant

	# Generate Initial Conditions
	cp.random.seed(17)            # set the random number generator seed
	
	mass = 20.0*cp.ones((N,1))/N  # total mass of particles is 20
	pos  = cp.random.randn(N,3)   # randomly selected positions and velocities
	vel  = cp.random.randn(N,3)
	
	# Convert to Center-of-Mass frame
	vel -= cp.mean(mass * vel,0) / cp.mean(mass)
	
	# calculate initial gravitational accelerations
	acc = get_acc( pos, mass, G, softening )
	
	# number of timesteps
	Nt = int(cp.ceil(tEnd/dt))
	
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
		
		# update time
		t += dt

	dt = tEnd - Nt
	vel += acc * dt/2.0
	pos += vel * dt
	acc = get_acc( pos, mass, G, softening )
	vel += acc * dt/2.0
	
	cp.cuda.Stream.null.synchronize()
	
	return 0
  
if __name__== "__main__":
        parser = argparse.ArgumentParser(description="Config file")
        parser.add_argument('--end_time', type=float, default=10, help='The end time used for simulation')
    
        args = parser.parse_args()  
        
        s = time.time()
        main(args.end_time)
        e = time.time()
        print(e-s)
