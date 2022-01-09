import MCMC
import numpy as np
import matplotlib.pyplot as plt

# Define some fitness function
def gaussian_posterior_func(x, mu, sigma):
	return np.exp(-(x-mu)**2/2/sigma**2)

gaussian_posterior = lambda x: gaussian_posterior_func(x, 2, 0.6)

# Define some schedule for the sampler
schedule = {"num_epochs_equilibration": 100,
			"num_epochs_sampling": 10000}

# Initialize an instance of the MCMC sampler
mcmc_sampler = MCMC.Sampler(fitness = gaussian_posterior,
							transition_sigmas = [0.2],
							theta_initial=[0])


# Run the sampler according to the schedule
mcmc_sampler.run(schedule=schedule)


# Plot the sampled data
plt.figure
t = np.linspace(-5, 5, 1000)
plt.plot(mcmc_sampler._theta_sequence_matrix,mcmc_sampler._fitness_sequence_matrix, 'bo')
plt.plot(t, gaussian_posterior(t), 'r-')
plt.show()

# Predict on identity function
mcmc_sampler.theta_stats()
