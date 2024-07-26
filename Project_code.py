import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import PillowWriter
from matplotlib.animation import FuncAnimation
from copy import deepcopy
import time

# Constants for the Plummer model
a = 1.0  # Scale length
R_v = 1.0  # Scale radius
N = 4
dt = 0.01
G = 4*np.pi**2
theta = 0.5
max_depth = 500
eps = 0.1

class Particle:
    '''
    This class represents the individual particles,
    A particle has three properties:
        - self.position = np.array([x,y]):
            A numpy array containing a list of the x and y 
            position of the particle. 
            units: see note

        - self.velocity = np.array([vx,vy]):
            A numpy array containing a list of the velocity of the particle
            in the x and y direction.
            units: see note

        - self.mass = float():
            A float representing the mass of the particle
            The mass itself is at the moment totaly unitless
            units: see note
    
    The class has three methods:    
        - self.update_position(self, position):
        - self.update_speed(self, velocity):
        - self.__repr__

    Note: The units are dependent on the choice of G, 
          during testing i've set G to 1. 
    '''
    def __init__(self, position, velocity, mass):
        '''
        Initializing the class
        inputs:
            position = np.array([x,y])
            velocity = np.array([vx,vy])
            mass     = float
        see the docstrings on the class for further explaination.
        '''
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = mass

    def update_position(self, position):
        '''
        This method is used to update the position to a new position
        inputs:
            position = same type as particle.position
            represents the new position of the particlE
        '''
        self.position = position #  just reset the self.position part
    
    def update_speed(self, velocity):
        '''
        This method is used to update the velocity to a new velocity
        inputs:
            velocity = same type as particle.velocity
            represents the new velocity of the particle'''
        self.velocity = velocity # again just a quick reset
        
    def __repr__(self):
        '''
        This method is used to be able to easily print out the information
        the Particle contains. 
        I used this in a sanity check, i got this from some stackexchange post
        and decided to use i myself

        '''
        return f"Particle(position={self.position}, velocity={self.velocity}, mass={self.mass})"

class QuadTreeNode:
    '''
    This class represents the individual Nodes of the tree,
    A single Node has seven properties:

        -self.bbox = [x_min, x_max, y_min, y_max]
            The boundary box of the node
            here [x_min, x_max, y_min, y_max], is a list
            of floats representing the corners of the box.

        -self.children = None
            An integer representing the amount of children a given Node has

        - self.total_mass = 0
            A float representing the total mass contained inside the Node

        - self.center_of_mass = np.zeros(2)
            A numpy array of 2 floats, representing
            the coords of the centre of mass
        
        - self.body = None
            (position, mass), representing the particle inside the bottom Nodes.
            A node can only contain a particle if it has no children!
            This part i wrote before i made the particle class so
            it's outdated but it works!
        
        - self.depth = depth
            Integer representing the amount of parents a Node has*
            Note that it doesn't keep track of whom the parents are.
            this is just here to make sure there is no infinite loop
        
        - self.max_depth = max_depth
            Integer representing the max depth a Node can have.
            It would make more sense to make this a property of the entire tree
            but i coded it up like this and it doesn't hurt.
    
    The class has four methods:    
        - self.update_center_of_mass(self, position, mass):
        - self.insert(self, position, mass):
        - self.subdivide(self):
        - self.compute_forces(self, particle, theta, G, eps):

    Note: The units are dependent on the choice of G, 
          during testing i've set G to 1. 
    '''
    def __init__(self, x_min, x_max, y_min, y_max, depth=0, max_depth=100):
        '''
        Initializing the class
        inputs:
            x_min     =     (float) Represents the minimum x coord of the boundary box
            x_max     =     (float)Represents the maximum x coord of the boundary box
            y_min     =     (float)Represents the minimum y coord of the boundary box
            y_max     =     (float)Represents the maximum y coord of the boundary box
            depth     = 0   (int)   Represents the current depth
            max_depth = 100 (int)   Represents the Maximum depth
        
        see the docstrings on the class for further explaination.
        '''
        self.bbox = [x_min, x_max, y_min, y_max] #set de boundary box
        self.children = None #a new Node has no children
        self.total_mass = 0 # and no mass 
        self.center_of_mass = np.zeros(2) # and no COM
        self.body = None # there is no particle in a new Node
        self.depth = depth # adding the current depth
        self.max_depth = max_depth # and the maximum possible depth

    def update_center_of_mass(self, position, mass):
        '''
        This method updates the Centre of mass 
        inputs:
            position = np.array([x,y]) representing the position of the particle
            mass     = float representing the mass of the particle
        
        '''
        self.center_of_mass = (self.center_of_mass * 
                               self.total_mass + position *
                                 mass) / (self.total_mass
                                           + mass) # we update the COM
        self.total_mass += mass # and add the added mass to the total mass

    def insert(self, position, mass):
        '''
        This function inserts a new particle in our Node
        inputs: 
            position = np.array([x,y]) representing the position of the particle
            mass     = float representing the mass of the particle
        
        ''' 
        if self.total_mass == 0: #if there's no mass we just update 
            self.total_mass = mass # the total mass
            self.center_of_mass = position # the COM
            self.body = (position, mass) # and the body
        elif self.children is None: # if there is already one or more particles:
            if self.depth >= self.max_depth: # we need to check if we are at our
                                            #recursion limit
                self.update_center_of_mass(position, mass) # if so: we 
                                            # treat the particles as one
            else: # if not we subdivide the Node
                self.subdivide() #subdivide
                quadrant = get_quadrant(self.bbox, self.body[0]) # in which quadrant
                                                        #do i need to add my particle
                self.children[quadrant].insert(self.body[0], self.body[1])#add the 
                                                                #original particle
                self.body = None # update the body since the Node now has children
                quadrant = get_quadrant(self.bbox, position) #where do i 
                                                            #need to add my new particle
                self.children[quadrant].insert(position, mass) # try adding the new particle
                self.update_center_of_mass(position, mass) # update the COM again
        else: # lastly, if we have a particle and the node already has children
            quadrant = get_quadrant(self.bbox, position) # find the child 
                                                        #the new particle belongs to
            self.children[quadrant].insert(position, mass) # and insert it there
            self.update_center_of_mass(position, mass) # update the COM

    def subdivide(self):
        '''
        This function subdivides a Node into four children
        and then updates the original Node to reflect this change
        '''
        x_min, x_max, y_min, y_max = self.bbox # given the bbox
        x_mid = (x_min + x_max) / 2 # we find the midpoints
        y_mid = (y_min + y_max) / 2 # of the x and y coords

        self.children = [
            QuadTreeNode(x_min, x_mid, y_mid, y_max, self.depth + 1, self.max_depth),  # Upper Left
            QuadTreeNode(x_mid, x_max, y_mid, y_max, self.depth + 1, self.max_depth),  # Upper Right
            QuadTreeNode(x_min, x_mid, y_min, y_mid, self.depth + 1, self.max_depth),  # Lower Left
            QuadTreeNode(x_mid, x_max, y_min, y_mid, self.depth + 1, self.max_depth)   # Lower Right
        ] 
        # and we update self.children with the four nodes
        # we make the children by creating four new nodes with boundary boxes
        # that divide the original node into four equal pieces. 
        # we increase the self.depth by one but not the max depth.

    def compute_forces(self, particle, theta, G, eps):
        '''
        This function calculates the forces acting on a Particle object
        inputs:
            particle =       Particle class representing the particle the force acts on
            theta    =       float  representing the resolution of our BH simulation
            G        =       float  representing the gravitational constant
            eps      =       float  representing an Error we introduce to make sure we don't divide by zero
        returns:
        np.array(2) containg the force in the x and y direction.


        '''
        force = np.zeros(2) # the force is set to zero at first
        position, mass = particle.position, particle.mass #define mass and pos
        
        if self.total_mass == 0 or (self.body is not None and np.array_equal(self.body[0], position)):
            return force # we return zero if the box is empty or if we found ourself
        

        L = self.bbox[1] - self.bbox[0] # setting L and r
        r = np.linalg.norm(self.center_of_mass - position)

        if L / r < theta: # the BH criteria, if this is the case we treat the box 
                        # as one big particle
            direction = self.center_of_mass - position #distance between the bodies
            distance = np.linalg.norm(direction) #from x,y to r
            force_magnitude = G * self.total_mass * mass / (distance**2 + eps**2)
                                # we calc the total force, eps**2 could also be just eps
            force += force_magnitude * direction / distance # we update the force in the
                                                        #x and y direction
                                                        # we need to divide by r
                                                        # due to vector reasons.

        elif self.children is not None: # if there are children and we cannnot assume 
                                    # the entire node is one big particle
            for child in self.children: # we just calculate the forces for all children
                                    # and add those
                force += child.compute_forces(particle, theta, G, eps)
        elif self.body is not None: # if the node in question has only one
                                    #if we reach the end we
                                    # calculate the forces on the two particles directly
            direction = self.body[0] - position
            distance = np.linalg.norm(direction)
            force_magnitude = G * self.body[1] * mass / (distance**2 + eps**2)
            force += force_magnitude * direction / distance

        return force

class QuadTree:
    '''
    This class represents the entire tree.
    the tree has 4 properties
        - self.particles = particles
            a list of Partciles object we are trying to simulate
        - self.theta = theta
            a float representing the resolution of our simulation
        - self.max_depth = max_depth
            a integer representing the recursion limit of the tree
        - self.root = QuadTreeNode(x_min, x_max, y_min, y_max, depth=0, max_depth=max_depth)
            The upper most Node, the root of the tree. 
    
    The class has nine methods:    
        - self.insert(self):
        - self.insert_new_particle(self, particle):
        - self.compute_all_forces(self, G, eps):
        - self.compute_total_energy(self, G, eps):
        - self.reconstruct_tree(self, new_particles):
        - self.leapfrog(self, dt, G, eps):
        - self.RK4(self , dt, G, eps):
        - self.copy(self):
        - self.RK4_adaptive(self , dt, G, eps, error_want):



    Note: The units are dependent on the choice of G, 
          during testing i've set G to 1. 
    '''

    def __init__(self, particles, theta, max_depth=100):
        '''
        Initializing the class
        inputs:
            particles =     (list)  List containing all Particle objects to simulate
            theta     =     (float) BH resolution of our tree  
            max_depth = 100 (int)   Represents the max recursion of the tree

        see the docstrings on the class for further explaination.
        '''
        self.particles = particles # all the particles in our tree
        self.theta = theta #set Resolution
        self.max_depth = max_depth # and max depth
        x_min = min(p.position[0] for p in particles) # we define the
        x_max = max(p.position[0] for p in particles) # bbox of the root
        y_min = min(p.position[1] for p in particles) # Node in such a way
        y_max = max(p.position[1] for p in particles) # that it will always 
                                                #contain all particles
        self.root = QuadTreeNode(x_min, x_max, y_min, y_max, depth=0, max_depth=max_depth)
        # and then set the root
    def insert(self):
        '''
        This function inserts all particles in self.particles in the tree
        by calling the insert function of the root Node
        '''
        for particle in self.particles: # for all particles
            self.root.insert(particle.position, particle.mass) #insert the particle

    def insert_new_particle(self, particle):
        '''
        This function adds a new particle given we already have a functioning tree
        inputs:
        particle =       Particle object we want to insert
        '''
        self.particles.append(particle) # we our particle to the 
                                    #list containing all particles
        self.reconstruct_tree(self.particles) # and then we reconstruct the tree
        # we can't just call the insert function because we need to update the
        # bbox of the root Node
            
    def compute_all_forces(self, G, eps):
        '''
        This function calculates all the forces acting on the particles 
        in self.particles
        inputs:
            G   =     float  the gravitational constant
            eps =     float  the Error avoiding term that makes sure we don't
                             divide by zero
        returns:
        np.array(N, 2) containing all the forces in the x and y direction
        for all the particles in self.particles
        '''
        forces = [] # setting empty list
        for particle in self.particles: #iterating over all particles
            force = self.root.compute_forces(particle, self.theta, G, eps)
            # for each particle we calculate the force it experiences by using
            # the compute_forces method on the Root Node
            forces.append(force) # we append the list
        return np.array(forces) # return the list

    def compute_total_energy(self, G, eps):
        '''
        This function calculates the total Energy of our system 
        inputs:
            G   =     float  the gravitational constant
            eps =     float  the Error avoiding term that makes sure we don't
                             divide by zero
        returns:
        float() representing the total energy
        
        '''
        # Compute kinetic energy per particle
        kinetic_energy = sum(0.5 * p.mass * np.linalg.norm(p.velocity)**2 for p in self.particles)
        
        # Compute potential energy by direct summation,
        # for our energy conversation checks we don't want an estimate. 
        potential_energy = 0 # set a counter
        for i, particle in enumerate(self.particles): # double for loop
            for j, other_particle in enumerate(self.particles):
                if i != j: # if we have 2 different particles:
                    distance = np.linalg.norm(particle.position - 
                                              other_particle.position) #calc r
                    potential_energy -= G * particle.mass * other_particle.mass / (distance + eps)
                    # calc potential E and subtract it from our total 
        potential_energy *= 0.5  # Avoid double-counting

        return kinetic_energy + potential_energy # return our total E
    
    def reconstruct_tree(self, new_particles):
        '''
        This function reconstructs the tree. 
        it is very similar to the __init__ function but it 
        also auto inserts all particle.
        I did it this way because i didn't know i could just call __init__
        but when i did i thought it would be nice to have an explicit reset button for
        clarity.

        inputs:
            new_particles =     list containing all the new Particles objects
                                in the tree

        '''
        theta = self.theta #we recall the self.root in a bit and we need theta
                        #as an argument
        max_depth = self.max_depth # same for max_depth
        self.root = None # reset self.root
        self.particles = new_particles # update the particles
        x_min = min(p.position[0] for p in self.particles)
        x_max = max(p.position[0] for p in self.particles)
        y_min = min(p.position[1] for p in self.particles)
        y_max = max(p.position[1] for p in self.particles)
        # again defince the bbox of the root node by making sure it contains all
        # particles
        self.root = QuadTreeNode(x_min, x_max, y_min, y_max, depth=0, max_depth=max_depth)
        # remake the root Node
        self.insert() # fill the tree again.

    def leapfrog(self, dt, G, eps):
        '''
        This function performs the leapfrog integration and updates the tree with
        the new positions and velocity's of the particles. 

        inputs:
            dt  =     float  the time interval
            G   =     float  the gravitational constant
            eps =     float  the Error avoiding term that makes sure we don't
                             divide by zero

        I used the Wikipedia page on frogleap integration as reference:
        https://en.wikipedia.org/wiki/Leapfrog_integration
        '''
        masses = np.array([p.mass for p in self.particles]) # array of all masses
        positions = np.array([p.position for p in self.particles])# array of all pos
        velocities = np.array([p.velocity for p in self.particles])# array of all velocities
        
        forces = self.compute_all_forces(G, eps) #np.array(2,N) of all forces
        accelerations = forces / masses[:, None] # getting acceleration
                    #we are trying to divide an array(2,N) by an array(N,)
                    # this will not work directly so we need to add
                    #[:, None], this transforms the mass array
                    # to shape array(N,1)

        
        half_step_velocities = velocities + 0.5 * accelerations * dt
        # calculate speeds at V_i+1/2
        new_positions = positions + half_step_velocities * dt
        # calculate the pos at i
    
        self.reconstruct_tree([Particle(pos, vel, mass) for pos, vel, mass in zip(new_positions, half_step_velocities, masses)])
        # reconstruct the tree with the new positions,
        # we use this to calculate the speeds at i+1
        #note we don't really care for the speeds at v_i+1/2, but we need to fill them in
        new_forces = self.compute_all_forces(G, eps) # calculate the new forces 
        new_accelerations = new_forces / masses[:, None] # and again calculate the acceleration 
                                               #using the same trick as before
        new_velocities = half_step_velocities + 0.5 * new_accelerations * dt
        # now we calculate the speeds at i+1
        self.reconstruct_tree([Particle(pos, vel, mass) for pos, vel, mass in zip(new_positions, new_velocities, masses)])
        # and reconstruct the tree using the right particles

    def RK4(self , dt, G, eps):
        '''
        This function performs the Runge-kutta 4 integration and updates the tree with
        the new positions and velocity's of the particles. 

        inputs:
            dt  =     float  the time interval
            G   =     float  the gravitational constant
            eps =     float  the Error avoiding term that makes sure we don't
                             divide by zero

        I used this site for reference since they had a nice worked out example for Newton's laws of motion:
        https://www.compadre.org/PICUP/resources/Numerical-Integration/

        I could probably use a for loop for generating K1 to K4 but i had a hard time finding
        my way around the half time steps so i didn't see that up untill i finished the code.
        I would be a lot cleaner to do so, but it wouldn't improve the performance.
        '''
        masses = np.array([p.mass for p in self.particles]) # array of all masses
        positions = np.array([p.position for p in self.particles])# array of all pos
        velocities = np.array([p.velocity for p in self.particles])# array of all velocities
        forces = self.compute_all_forces(G, eps) #np.array(2,N) of all forces
        accelerations = forces / masses[:, None] # same trick as in leapfrog, [:,None] reshapes the masses to the correct shape
        K1v = accelerations * dt #defining K1x and K1v
        K1x = velocities*dt
        new_pos_1 = positions + K1x/2 #update the positions and speeds
        new_vel_1 = velocities + K1v/2
        self.reconstruct_tree([Particle(pos, vel, mass) for pos, vel, mass in zip(new_pos_1,new_vel_1, masses)]) #reconstruct tree
        forces_at_half = self.compute_all_forces(G, eps) #np.array(2,N) of all forces 
        accelerations_at_half = forces_at_half / masses[:, None] #recalc acceleration
        K2v = accelerations_at_half*dt #K2x and K2v
        K2x = (velocities + K1v/2)*dt
        new_pos_2 = positions + K2x/2 #again new positions
        new_vel_2 = velocities + K2v/2 # and speeds
        self.reconstruct_tree([Particle(pos, vel, mass) for pos, vel, mass in zip(new_pos_2,new_vel_2, masses)]) #reconstruct
        forces_at_half = self.compute_all_forces(G, eps) #np.array(2,N) of all forces
        accelerations_at_half = forces_at_half / masses[:, None] # a for calculating K3
        K3v = accelerations_at_half*dt #k3x and K3v
        K3x = (velocities + K2v/2)*dt
        new_pos_3 = positions + K3x # again new pos and speeds
        new_vel_3 = velocities + K3v
        self.reconstruct_tree([Particle(pos, vel, mass) for pos, vel, mass in zip(new_pos_3,new_vel_3, masses)]) # reconstruct
        forces_at_half = self.compute_all_forces(G, eps) #np.array(2,N) of all forces
        accelerations_at_half = forces_at_half / masses[:, None] # for K4
        K4v = accelerations_at_half*dt #k4x and v
        K4x = (velocities + K3v)*dt 
        true_new_v = velocities + (K1v + 2*K2v + 2*K3v + K4v)/6 # all put together we get the expression for x_i+1 and v_1+1
        true_new_pos = positions + (K1x + 2*K2x + 2*K3x + K4x)/6
        self.reconstruct_tree([Particle(pos, vel, mass) for pos, vel, mass in zip(true_new_pos,true_new_v, masses)]) # and reconstruct the tree accordingly

    def copy(self):
        '''
        returns a copy of itself
        '''
        return deepcopy(self)
    
    
    def RK4_adaptive(self, dt, G, eps, error_want):
        '''
        This function performs the Runge-kutta 4 integration using adaptive timesteps and updates the tree with
        the new positions and velocity's of the particles. It also returns a new timestep to use.

        inputs:
            dt  =     float  the time interval
            G   =     float  the gravitational constant
            eps =     float  the Error avoiding term that makes sure we don't
                             divide by zero
            error_want = the error size we want to get


        I Think i've implemented this wrong; i did this hurried. 
        The problem problem lies somewhere in the if statement;
        due to time contrains and me not notacing it untill after writing the
        rapport/testing it against the regular one i will not be fixing it.
        
        i think removing the > 0 part and only checking if 
        relative_eror <= error_want could do the trick

        '''
        y_start = self.copy() # we store the initial tree
        self.RK4(dt, G, eps) # then perform a single timestep
        ysingle = self.copy() # and copy it as ysingle
        dt_half = 0.5 * dt # we half the timesteps
        y_start.RK4(dt_half, G, eps) #we then use the y_start tree (original tree) to perform the double integration
        y_start.RK4(dt_half, G, eps)# in succesion
        ydouble = y_start.copy() # and we copy it also

        particles_single = np.asarray([p.position for p in ysingle.particles])# we then check the error
        particles_double = np.asarray([p.position for p in ydouble.particles])#by using particles position
        
        relative_error = np.linalg.norm((particles_double - particles_single) / (particles_double + 1e-10))  # Prevent division by zero
        if relative_error > 0:  # Prevent division by zero issues
            dt_new = dt * (error_want / relative_error)**0.2
        else:
            dt_new = dt  # If error is zero, keep the time step unchanged

        if relative_error <= error_want:
            self.reconstruct_tree([Particle(p.position, p.velocity, p.mass) for p in ydouble.particles])
        else:
            dt_new = 0.5 * dt

        return dt_new

def get_quadrant(bbox, position):
    '''
    This function checks in which child of the current bbox the particle is in
    and then return the list index of that child
    '''
    x_min, x_max, y_min, y_max = bbox # unpack the bbox
    x, y = position # unpack the position
    x_mid = (x_min + x_max) / 2 #calc midpoint of the bbox x
    y_mid = (y_min + y_max) / 2 # and y
    # the subdivide function creates the children
    # from top to bottom from left to right,
    # this we will abuse by returning only the 
    # index of the box the particle is in
    if x < x_mid and y >= y_mid: 
        return 0  # Upper Left
    elif x >= x_mid and y >= y_mid:
        return 1  # Upper Right
    elif x < x_mid and y < y_mid:
        return 2  # Lower Left
    else:
        return 3  # Lower Right

def initialize_particles(N, mass, Speed_type = None, G = None, pos_mass = None):
    '''
    This function generates N random particles
    inputs: 
        N          =       integer  how many particles we want
        mass       =       float    representing the mass of the particles
        Speed_type = None  str      can also be 'Rotational'
                                    decides which method to generate the speed
                                    with
    
    '''
    mass_array = np.ones_like(N) * mass
    radii = IPDF_plummer_model(N) # create Radii
    mass_array = np.ones_like(radii) * mass
    if pos_mass != None:
        mass_array = np.append(mass_array,pos_mass[1])
        radii = np.append(radii, pos_mass[0])
    x, y = conversion_r_to_xy(radii) # convert them to x and y
    if Speed_type == None: # if we don't care which method to use
        vx, vy = generate_speed_xy(N) # we create random speeds
    if Speed_type == 'Rotational': # if we use the rotational one
        vx, vy = generate_speed_xy(N) # we create the random speeds
        theta = np.arctan2(x,y) # but use the method described 
        vx = vx * -np.sin(theta) # in the assignment
        vy = vy *np.cos(theta) # in addition
    if Speed_type == 'Differential':

       v_t = calculate_rotational_speeds(radii, mass_array, G) 
       vx, vy = calculate_velocity_components(v_t, x, y)
        
    # we now create a list full of Partciles objects assigned with our created 
    #positions and speeds using zipped lists.
    particles = [Particle(position, velocity, mass) for position, velocity, mass in zip(zip(x, y), zip(vx, vy), mass_array)]
    return particles #return our list

def IPDF_plummer_model(N=1):
    '''
    This function generates N random radii according to the plummer_model
    described in the assignment
    '''
    X = np.random.uniform(low=0.0, high=1.0, size=N)  #N samples from U(0,1)
    denum = np.sqrt((X**(-2/3)-1)) # denumenator
    p_bar = R_v/denum # calculate the density
    return p_bar # return the radius given the density

def conversion_r_to_xy(r):
    '''
    This function converts the radius into an x and y coordinate
    It does this by randomly selecting two different angles
    theta and phi and using the method described by the assignment
    to calculate the x and y coordinates
    inputs: 
        r =     list(N) containing all the radii of the particles
    returns:
        x =     list(N) containing all x values
        y =     list(N) containing all y values
    '''
    phi = np.random.uniform(low=0.0, high=1.0, size=len(r)) #phi~U(0,1)
    theta = np.random.uniform(low=0.0, high=1.0, size=len(r))#same for theta
    x = np.asarray(r)*np.sin(np.arccos((2*theta)-1))*np.cos(np.pi*2*phi) #calc x
    y = np.asarray(r)*np.sin(np.arccos((2*theta)-1))*np.sin(np.pi*2*phi)# and y
    return x, y # return

def generate_speed_xy(N):
    '''
    This function generates the speed in the x and y direction
    using the method described in the assignment
    inputs:
        N =    int how many speeds to generate
    returns
        vx =   list(N) containing all x values
        vy =   list(N) containing all x values
    '''
    v = 0.5*np.sqrt(2)  # velocity dispersion
    n_1 = np.random.normal(1, size=N)
    n_2 = np.random.normal(1, size=N)
    return v*n_1, v*n_2

def collect_bboxes(node, bboxes):
    '''
    This method appends all the boxes contained by the tree
    to the bboxes input list
    input:
        node   =     QuadtreeNode for which we want the box
        bboxes =     A list containing all the previously added boxes

        note: to run this code you first need to make an emtpy list
            then you call this function on the root Node of the tree     
    '''
    bboxes.append(node.bbox) # we append the current bbox
    if node.children: # and if the Node has children we rerun the function
        for child in node.children:
            if child:
                collect_bboxes(child, bboxes)

def scatter_particles_and_bboxes(quadtree):
    '''
    This function creates a scatter plot of 
    the particles and the tree(represented as boxes)
    inputs: 
        quadtree = Quadtree object to make a scatterplot of
    
    
    '''
    fig, ax = plt.subplots(figsize=(10, 10)) # create figure
    particles = quadtree.particles # get all the particles
    positions = np.array([p.position for p in particles]) # and pos of particles
    ax.scatter(positions[:, 0], positions[:, 1], s=1, color='blue') # scatter them
    
    bboxes = [] # create an empty list for the collect_boxes function
    collect_bboxes(quadtree.root, bboxes) # and collect all the boxes of the tree
    for bbox in bboxes: # now for every box we create a patch in the figure
        x_min, x_max, y_min, y_max = bbox
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=0.1, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
    
    ax.set_aspect('equal')# we create nice labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('QuadTree Bounding Boxes and Particle Scatter Plot')
    plt.show()

def calculate_forces_by_hand(particles, G, eps):
    '''
    This function calculates the forces acting on the particles one by one.
    this was used to double check the tree method
    inputs:
        particles = list of Particles objects we want the forces for
        G   = float gravitational constant
        eps = floa that makes sure we don't divide by zero
    '''
    force_byhand = np.zeros((len(particles), 2)) # we make an empty array
    for i, particle in enumerate(particles): # double for loop
        for j, particle2 in enumerate(particles):
            if i != j: # makes sure have different particles
                pos = particle.position 
                mass = particle.mass
                pos2 = particle2.position
                mass2 = particle2.mass
                norm = np.linalg.norm(pos - pos2) # calc r
                num = G * mass * mass2 * (pos2 - pos) # the numerator
                denum = (norm**3 + eps**2) # denominator
                force_byhand[i] += num / denum # calc
    return force_byhand # return

def insert_particle(pos, velocity, mass):
    '''
    this funciton allows us to loop over the Quadtree,insert_new_particles()
    function
    ...
    '''
    P = Particle(pos, velocity, mass)
    quadtree.insert_new_particle(P)

def get_animate_particles(quadtree, type= None, G=1.0, eps=0.001, dt=0.01, with_boxes = False):
    '''
    This functions returns a function that can be passed into the FuncAnimation function from 
    matplotlib

    inputs:

    quadtree   =         Quadtree    The tree we want to animate
    type       = None    str         the integrator to use None;frog-leap,
                                     'RK4' or 'RK4 adaptive'
    G          = 1.0     float       The gravitation constant
    eps        = 0.001   float       the smoothin parameter
    dt         = 0.01    float       timestep
    with_boxes = False   Bool        False->no boxes visible, True -> boxes visible   
    outputs:
    animate_particles(j) function we can use to create the animations.  
    
    '''
    errors = [] # list of errors
    E_start = quadtree.compute_total_energy(G, eps) #starting energy
    timelist = [] # for plotting the dt changing 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 1]})
    def animate_particles(j):
        nonlocal dt # this does some python magic allowing dt to change every time we call the function
        ax1.clear() # we clear the previous plots
        ax2.clear()
        
        # and recreate plots
        particles = quadtree.particles # we want to plot the particles
        positions = np.array([p.position for p in particles])
        ax1.scatter(positions[:, 0], positions[:, 1], s=10, color='black')
        ax1.grid() # setting some layout stuff
        ax1.set_xlabel('X-axis in AU')
        ax1.set_ylabel('Y-axis in AU')
        ax1.set_title('N-Body simulation using the BarnesHut algorithm')
        ax1.set_xlim(-10, 10)
        ax1.set_ylim(-10, 10)
        if with_boxes == True:  # if we want to show the boxes
            boxes = [] # we add the boxes by calling the collect boxes funciton
            collect_bboxes(quadtree.root, boxes)
            for box in boxes:
                x_min, x_max, y_min, y_max = box # and we plot each
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
                ax1.add_patch(rect)
            ax1.set_xlim(min(boxes, key=lambda x: x[0])[0], max(boxes, key=lambda x: x[1])[1])
            ax1.set_ylim(min(boxes, key=lambda x: x[2])[2], max(boxes, key=lambda x: x[3])[3])
        #we use either RK4, RK4 adaptive or if not specified leapfrog to update the quadtree
        if type == 'RK4':
            quadtree.RK4(dt, G, eps)
        if type == 'RK4 adaptive': # here we also add dt to the errorplot
            # so we can see how dt changes.
            dt = quadtree.RK4_adaptive(dt, G, eps,0.001)
            timelist.append(dt)
            ax2.plot(timelist, label = 'dt')
        else:
            quadtree.leapfrog(dt, G, eps)
        # Then we plot the error
        E = quadtree.compute_total_energy(G, eps) # we get the energy right now
        error = abs(1 - E / E_start) # relative error
        if error != 0: # if zero -> log scale is angry
            errors.append(error) # we append the error
        else:
            erros.append(1e-6) # placeholder for zero. 
            #i chose 1e-6 because if we choose a lower value we loos detail in the area of interest
        # setting the layout for the error plot
        ax2.plot(errors, color='red', label = 'error')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Energy Error')
        ax2.set_xlim(0, len(errors))
        ax2.set_ylim(min(errors), max(errors)+1 if errors else 1)
        ax2.set_yscale('log')
        ax2.grid()
        ax2.legend()
        plt.tight_layout()

    return animate_particles # return this function to iterate over

def calculate_rotational_speeds(radius_array, mass_array, G):
    '''
    This function calculates the rotational (tangetal) velocity of an array of particles

    inputs:

    radius_array =         list[N] of the radial position of N particles
    mass_array   =         list[N] of the mass of N particles
    G            =         float   gravitational constant.

    returns:

    np.array[N] of the tangetal velocity of the particles
    '''

    # Ensure both arrays are numpy arrays for easier manipulation
    mass_array = np.array(mass_array) # we need the lists as array to perform
    radius_array = np.array(radius_array) # the calculations
    tangental_speeds = [] # empty array
    
    for r in radius_array: # iterating over every r
        M = mass_array[r>radius_array].sum() # we sum over every mass that
                                            #has a lower radius than where we are at
        if r == 0: # we can't divide by zero
            tangental_speeds.append(0)
        else:
            tangental_speeds.append(np.sqrt(G * M / (r))) # we calculate the speeds
            # AFTER WRITING NOTE: i straight up forgot the +1e-... part
            # as r -> 0 we could excidentally blow up the speed.



    # Calculate the rotational speeds
    
    return np.asarray(tangental_speeds)

def Solar_system(G):
    '''
    This function initializes particles that represent the solar system

    not that for this to work: G has to be 4 * pi**2
    inputs: 
    G = float: grav. constant
    returns
    list containing Particles objects representing the solar system.

    Note: i used chat gpt to create this list for me, i've checked up to jupiter
    if it is accurate.
    '''
    # Planetary data: [distance in AU, mass in solar masses]
    planetary_data = [
        [0, 1],                     # Sun
        [0.39, 3.30e-7],            # Mercury
        [0.72, 2.45e-6],            # Venus
        [1.00, 3.00e-6],            # Earth
        [1.52, 3.21e-7],            # Mars
        [5.20, 9.54e-4],            # Jupiter
        [9.58, 2.86e-4],            # Saturn
        [19.22, 4.37e-5],           # Uranus
        [30.05, 5.15e-5],           # Neptune
        [39.48, 6.55e-9]            # Pluto
    ]

    radii = [data[0] for data in planetary_data] # we create a new list
    masses = [data[1] for data in planetary_data] # with the masses and radii
    # so we can use this in our differential speed calculation
    rotational_speeds = calculate_rotational_speeds(radii, masses, G)
    x, y = conversion_r_to_xy(radii) # we convert to cartesian coords
    # and do the same for our tangental speed
    vx, vy = calculate_velocity_components(rotational_speeds, x, y)
    # and then we return this as a list of Particles
    particles = [Particle(position, velocity, mass) for position, velocity, mass in zip(zip(x, y), zip(vx, vy), masses)]
    return particles 

def calculate_velocity_components(rotational_speeds, x, y):
    '''
    This function converts the tangental speed to a speed in the x and y direction

    inputs:
    rotational_speeds = np.array(N) containing the tangental speeds
    x = np.array(N) containing the x posisitons of the particles
    y = np.array(N) containing the y posisitons of the particles

    '''
    r = np.sqrt(x**2 + y**2) # we need the radius
    # i did this instead of using r as an input because it allows me to also use this
    #in our Initialize particle function

    # the velocity components are defined in the assignment but is just
    # times an sin / cos of the angle
    # which i did by explicitly writing it as a ratio 
    vx = -rotational_speeds * y / (r + 1e-7) # this is not the smoothin parameter!
    vy = rotational_speeds * x / (r + 1e-7) # same here, it just so that if we place an object at zero zero we don't divide by zero
                                            # also the smoothin parameter is specifically for the forces
    
    return vx, vy 

#particles = initialize_particles(N, 0.01,Speed_type= 'Differential', G = G, pos_mass =[0, 10])
particles = Solar_system(G)
quadtree = QuadTree(particles, theta, max_depth)
quadtree.insert()
insert_particle([-3,3], [0,1], 1e-16) # adding a comet


animate_particles = get_animate_particles(quadtree,'', G, eps, dt)
ani = FuncAnimation(plt.gcf(), animate_particles, interval=1, frames= 1000, repeat=False)
writergif = PillowWriter(fps=30)
ani.save('filename.gif',writer=writergif)
print('done')

    # what i need to do:
    # make the nice plots
    # add fun stuff
    # Beeman algorithm - > mention in rapport, not self starting
    # note that for the leapfrog and RK4 we do half a timestep that we defined with dt
    # so 2*dt is one timestep.
    # define all the constants we need and mention them!!!