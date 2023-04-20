import math
import numpy as np
from scipy.linalg import norm

from src.ex4_utils import sample_gauss
from src.motion_model import get_system_matrices
from src.ex2_utils import extract_histogram, get_patch, create_epanechnik_kernel
from src.tracker import Tracker
#from src.ex2_utils import Tracker

class ParticleParams():
    def __init__(self,
                 kernel_sigma: float = 0.5,
                 nbins: int = 24,
                 q: float = 200,
                 r: float = 1,
                 histogram_update_alpha: float = 0.2,
                 num_of_particles: int = 300,
                 motion_model: str = "nearly_constant_velocity",
                 sigma_squared: float = 0.01):
        self.num_of_particles = num_of_particles
        self.motion_model = motion_model
        self.nbins = nbins
        self.q = q
        self.r = r
        self.histogram_update_alpha = histogram_update_alpha
        self.sigma_squared = sigma_squared
        self.kernel_sigma = kernel_sigma

class ParticleTracker(Tracker):
    parameters: ParticleParams
    particles: np.array
    weight: np.array
    histogram: np.array

    def __init__(self, params: ParticleParams = ParticleParams()):
        self.parameters = params
        #super().__init__()

    def name(self):
        return "Particle_filter_tracker_kernel_05_particle_300_sigma_01_alpha_02"

    def initialize(self, image, region):

        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        # Round the region into integers
        region = [round(r) for r in region]

        # Make window size an odd number
        if(region[2]%2 == 0):
            region[2] = region[2]+1
        if(region[3]%2 == 0):
            region[3] = region[3]+1

        # Get center position and size of the window
        self.position = (round(region[0] + region[2] / 2), round(region[1] + region[3] / 2))
        self.size = (int(region[2]), int(region[3]))

        patch, mask = get_patch(image, self.position, self.size)
        # mask = np.repeat(mask.reshape((mask.shape[0], mask.shape[1], 1)), 3, axis=2)
        # patch = patch*mask

        #self.parameters.q = min(self.size[0],self.size[1])/5
        self.parameters.q = (patch.size/image.size)*400

        # Get matrices
        self.Fi, self.Q, self.H, self.R = get_system_matrices(self.parameters.motion_model,
                                                              self.parameters.q,
                                                              self.parameters.r)

        # Initialize particles
        # Create kernel as it will be used always
        self.kernel = create_epanechnik_kernel(self.size[0],self.size[1],
                                               self.parameters.kernel_sigma)
        self.histogram = extract_histogram(patch, self.parameters.nbins, self.kernel)
        self.histogram = self._normalize_histogram(self.histogram)
        self.particles = sample_gauss([self.position[0],self.position[1],0,0],
                                        self.Q,
                                        self.parameters.num_of_particles)
        # for point in self.particles:
        # #     point[0] = round(point[0])
        # #     point[1] = round(point[1])
        #      point[2] = 0
        #      point[3] = 0

        self.weights = np.array([1]* self.parameters.num_of_particles)
        self.normalized_weights = self.weights/sum(self.weights)

    def track(self, image):
        # Sample particles according to weight
        particles_new = self._sample_particles()
        # Use motion model to transform weights
        self.particles = self._transform_particles(particles_new)

        # Compute weights from visual model
        self._compute_weights(image)
        # Compute weighted sum with new weights
        new_state =  self._compute_new_state()
        self.position = (new_state[0], new_state[1])

        # Update_histogram
        new_patch, new_mask = get_patch(image, self.position, self.size)
        # new_mask = np.repeat(new_mask.reshape((new_mask.shape[0], new_mask.shape[1], 1)), 3, axis=2)
        # new_patch = new_patch*new_mask
        new_histogram = extract_histogram(new_patch, self.parameters.nbins, self.kernel)
        
        new_histogram = self._normalize_histogram(new_histogram)

        self.histogram = self.parameters.histogram_update_alpha*self.histogram + (1-self.parameters.histogram_update_alpha)*new_histogram

        return [self.position[0]-(self.size[0]/2), self.position[1]-(self.size[1]/2), self.size[0], self.size[1]]

    def _sample_particles(self):
        weights_norm = self.weights/np.sum(self.weights) # normalize weights
        weights_cumsumed = np.cumsum(weights_norm) # cumulative distribution
        rand_samples = np.random.rand(self.parameters.num_of_particles, 1)
        sampled_idxs = np.digitize(rand_samples, weights_cumsumed) # randomly select N indices
        particles_new = self.particles[sampled_idxs.flatten(), : ]

        return particles_new

    def _compute_weights(self, image: np.array):
        self.weights = []

        for particle in self.particles:
            particle_patch, particle_mask = get_patch(image, (particle[0], particle[1]), self.size)
            if(particle_patch.shape != (self.size[1], self.size[0], 3)):
                breakpoint()
            # particle_mask = np.repeat(particle_mask.reshape((particle_mask.shape[0],
            #                                                  particle_mask.shape[1],
            #                                                  1)), 3, axis=2)
            # particle_patch = particle_patch*particle_mask
            particle_histogram = extract_histogram(particle_patch,
                                                   self.parameters.nbins,
                                                   self.kernel)
            particle_histogram = self._normalize_histogram(particle_histogram)
            
            d_hel = norm(np.sqrt(particle_histogram) - np.sqrt(self.histogram))/np.sqrt(2)

            w_i = math.exp(-(d_hel**2)/(2*self.parameters.sigma_squared))

            self.weights.append(w_i)
        
        self.weights = np.array(self.weights)
        
        self.normalized_weights = self.weights/sum(self.weights)

    def _transform_particles(self, particles_new: np.array):
        transformed = []

        noise = sample_gauss([0, 0, 0, 0], self.Q, self.parameters.num_of_particles)

        for particle in particles_new:
            transformed.append(self.Fi.dot(particle))
        return transformed + noise

    def _compute_new_state(self):
        weighted = [point*self.normalized_weights[i] for i, point in enumerate(self.particles)]
        summed = np.array(sum(weighted))
        rounded = np.round(summed)
        return rounded

    def _normalize_histogram(self, histogram: np.array) -> np.array:
        return histogram / np.linalg.norm(histogram)