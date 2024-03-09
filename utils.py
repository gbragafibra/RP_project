import numpy as np 

def kaiser(λ_list):
	"""
	Takes a list of eigenvalues
	Returns how many dimensions
	should be represented
	according to the Kaiser test
	"""

	PC = 0

	for i in range(λ_list.shape[0]):
		PC += 1
		if λ_list[i] < 1:
			break

	return PC


def scree(λ_list, ϵ):
	"""
	Takes a list of eigenvalues
	Returns how many dimensions
	should be represented
	according to the Scree test
	Also takes a threshold ϵ to
	compare to the difference
	between consecutive eigenvalues
	"""

	PC = 0

	for i in range(λ_list.shape[0]):
		PC += 1
		if abs(λ_list[i-1] - λ_list[i]) < ϵ:
			break

	return PC

def scale_σ(X):
	"""
	Takes a matrix and normalizes
	along cols, taking into account
	mean (μ) and std (σ)
	"""
	μ = np.mean(X, axis = 0)
	σ = np.std(X, axis = 0)
	normalized_X = (X - μ)/σ

	return normalized_X