import numpy as np

class Sampler:
	def __init__(self, fitness, theta_initial, transition_sigmas, random_seed=42):
		"""
		Initialize the class instance.
		"""
		# Assign inputs to class attributes
		self._fitness = fitness
		self._current_theta = np.array(theta_initial)
		self._transition_sigma = np.diag(transition_sigmas)
		self._random_seed = random_seed

		# Initialize the random seed
		np.random.seed(self._random_seed)

		# Initialize the list '_sampled_thetas' and '_sampled_fitnesses' as empty list 
		self._sampled_thetas    = []
		self._sampled_fitnesses = []

		# Evaluate the fitness for the current theta and assign it to a class attribute
		self._fitness_current_theta = self._fitness(self._current_theta)

	def _update(self, sampling=False):
		"""
		Update the theta value, while either storing the new theta (sampling=true)
		or not (sampling=false).
		"""
		# Propose a new theta by sampling from a gaussian distribution
		proposed_theta = np.random.multivariate_normal(self._current_theta, self._transition_sigma)

		# Evaluate the fitness function for the proposed theta
		fitness_proposed_theta = self._fitness(proposed_theta)	

		# Check if the proposed fitness is not 0
		if fitness_proposed_theta!=0:
			# Calculate the ratio between the proposed fitness and the current fitness
			ratio = fitness_proposed_theta/self._fitness_current_theta
			
			# Sample a random value in [0,1[ and check if it is smaller or equal than the ratio
			if np.random.rand()<=ratio:
				# Update the current theta 
				self._current_theta = proposed_theta

				# In this case the current fitness corresponds to the proposed fitness (after updating)
				self._fitness_current_theta = fitness_proposed_theta 

		# If sampling==True, add the current theta and fitness to the list of sampled thetas and fitnesses
		if sampling==True:
			self._sampled_thetas.append(self._current_theta)
			self._sampled_fitnesses.append(self._fitness_current_theta)

	def _equilibrate(self, num_epochs):
		"""
		Equlibrate the system for some number of epochs.
		"""
		
		print()
		print("Start the equilibration.")
		
		# Loop over the number of epochs
		for epoch in range(num_epochs):
			# Update the system without storing the thetas (sampling=False)
			self._update(sampling=False)

		print("Finished the equilibration.")

	def _sample_sequence(self, num_epochs):
		"""
		Sample a sequence of thetas for for some number of epochs.
		"""
	
		print()
		print("Start the sequence sampling.")

		# Loop over the number of epochs
		for epoch in range(num_epochs):
			# Update the system while storing the thetas (sampling=True)
			self._update(sampling=True)
				
		print("Finished the sequence sampling.")
		
	def _construct_sequence_matrix(self):
		"""
		Construct a 2dimension matrix (#sampled thetas, |theta|).
		"""
		# Check if the sequence list is not empty
		if len(self._sampled_thetas)==0:
			raise ValueError("There are not sampled thetas. Can't construct the sequence matrix.")

		# Stack the list of 1d arrays to get the 2d matrix
		self._sequence_matrix = np.vstack(self._sampled_thetas)
		
	def run(self, schedule):
		"""
		First equilibrate the system and then sample a sequence.
		"""
		print()
		print("Start")
		
		# 1) Equilibrate the system
		self._equilibrate(num_epochs = schedule["num_epochs_equilibration"])
		
		# 2) Sample a sequence
		self._sample_sequence(num_epochs = schedule["num_epochs_sampling"])

		# 3) Construct the sequence matrices
		self._construct_sequence_matrices()

		print()
		print("Finished.")

	def _construct_sequence_matrices(self):
		"""
		Construct 2dimension matrices of the thetas [shape (#sampled thetas, |theta|)]
		and of the fitnesses [shape (#samples thetas, 1)] for the sampled thetas and fitnesses.
		"""
		# Check if the sequence list is not empty
		if len(self._sampled_thetas)==0:
			raise ValueError("There are no sampled thetas. Can't construct the sequence matrices.")

		# Stack the lists of 1d arrays to get the 2d matrices
		self._theta_sequence_matrix   = np.vstack(self._sampled_thetas)
		self._fitness_sequence_matrix = np.vstack(self._sampled_fitnesses)

		# Calculate the sum over the fitness_sequence_matrix to obtain the normalizer of the sampled fitnesses
		fitness_sequence_normalizer = np.sum(self._fitness_sequence_matrix)

		# Normalize the fintess_sequence matrix
		self._fitness_sequence_matrix_normalized = self._fitness_sequence_matrix/fitness_sequence_normalizer

	def theta_stats(self):
		"""
		Get the mean and standard deviation of the thetas.
		"""
		# Get the mean of the sampled thetas
		theta_mean = np.mean(self._theta_sequence_matrix)

		# Get the covariance matrix of the sampled theta
		theta_cov = np.cov(self._theta_sequence_matrix.transpose())
		
		# Check if the covariance matrix is zero dimensional or not 
		if theta_cov.ndim == 0:
			# Get sqrt of the 0dim covariance matrix 
			theta_std = np.sqrt(theta_cov)
		else:
			# Get the square root of the diagonalized covariance matrix of the sampled theta
			theta_std = np.sqrt(np.linalg.eig(theta_cov))
	
		print("mean(theta) = ", theta_mean)
		print("cov(theta)  = ", theta_cov)
		print("std(theta)  = ", theta_std)


	def predict(self, predictor):
		"""
		Predict the average of the predictor function evaluated on the theta.
		"""
		# Evaluate the predictor on the theta sequence matrix, 
		# which should result in a matrix of size (#sampled thetas, 1)
		predictor_matrix = predictor(self._theta_sequence_matrix)

		# Calculate the mean of the predictor matrix
		prediction = np.mean(predictor_matrix)

		return prediction

