import numpy as np 
import pandas as pd 

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


def kept_var(λ_list, n):
	"""
	Takes a list of eigenvalues
	Computes the kept variance
	for n amount of features
	"""

	kept_var = 0 

	for i in range(0,n):
		kept_var += λ_list[i]**2

	return (kept_var/np.sum(λ_list**2))


def initial_processing(file):
	"""
	Initial processing for 
	the respective datafile
	Returns separated matrices
	of features, and vector
	with labels
	"""

	credit_df = pd.read_excel(file)
	"""
	Both remove the handles
	Also don't care about "feature" ID
	"""
	features = credit_df.iloc[1:, 1:-1].values.astype(float)
	labels = credit_df.iloc[1:, -1].values.astype(int)  

	return features, labels


def compute_λ_cov(X):
	"""
	Takes a matrix X
	and computes the cov matrix
	(normalizing it before)
	and respective eigvals and
	eigvecs
	"""
	cov_X = np.cov(scale_σ(X).T) #Need to transpose
	λ_vals, λ_vecs = np.linalg.eig(cov_X)

	return cov_X, λ_vals, λ_vecs
