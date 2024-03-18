import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from scipy.stats import kstest
"""
kstest more appropriate for
larger sample sizes
"""

def kaiser(λ_list):
	"""
	Takes a list of eigenvalues
	Returns how many dimensions
	should be represented
	according to the Kaiser test
	"""

	PC = 0

	"""
	Need to order λ_list (descending)
	"""
	λ_list = λ_list[::1]

	for i in range(λ_list.shape[0]):
		PC += 1
		if λ_list[i] < 1:
			break

	return PC


def scree(λ_list, ε):
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

	"""
	Need to order λ_list (descending)
	"""
	λ_list = λ_list[::1]

	for i in range(λ_list.shape[0]):
		PC += 1
		if abs(λ_list[i-1] - λ_list[i]) < ε:
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

	"""
	Need to order λ_list (descending)
	"""
	λ_list = λ_list[::1]

	kept_var = 0 

	for i in range(0,n):
		kept_var += λ_list[i]**2

	return (kept_var/np.sum(λ_list**2))


def initial_processing(file):
	"""
	Initial processing for 
	the respective datafile
	Returns separated matrices
	of features, vector
	with label vals, and handles for
	the features
	"""

	credit_df = pd.read_excel(file)
	"""
	Both remove the handles
	Also don't care about "feature" ID
	"""
	handles = credit_df.iloc[0, 1:-1]
	features = credit_df.iloc[1:, 1:-1].values.astype(float)
	labels = credit_df.iloc[1:, -1].values.astype(int)  

	return features, labels, handles


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


def kruskal_wallis(X, labels):
	"""
	For the case of binary labels
	Kruskal-Wallis test to rank
	features.
	Returns a vector with all scores
	for the respective features
	Takes X matrix with feature values
	and labels vector to acess classes.

	!Doesn't include tie discrimation yet!
	"""

	labels_stats = np.unique(labels, return_counts=True)
	Hs = []
	R_avg = np.sum(np.arange(1, X.shape[0] + 1))/X.shape[0] #avg rank
	for i in range(X.shape[1]):
		"""
		Initialize ranks regarding each class
		"""
		Rs = [0, 0]

		X_updated = np.column_stack((X[:, i],labels))
		"""
		Sort according to feature vals ([0])
		"""
		X_updated = X_updated[X_updated[:, 0].argsort()]
		
		for j in range(X.shape[0]):
			if X_updated[j,0] == labels_stats[0][0]:
				Rs[0] += j + 1
			else:
				Rs[1] += j + 1
		"""
		Getting the average ranks
		"""
		Rs[0] /= labels_stats[1][0]
		Rs[1] /= labels_stats[1][1]

		H = 0
		for k in range(len(labels_stats[0])):
			H += labels_stats[1][k] * (Rs[k] - R_avg)**2

		H *= 12/(X.shape[0] * (X.shape[0] + 1))
		Hs.append(H)
	
	return Hs


def redundancy_check(X, ε, labels):
	"""
	Checking for redundancy
	between features.
	Takes a matrix X, a 
	threshold ε (i.e. ε = 0.95)
	to possibly eliminate some feature.
	Such is done also taking into account
	the scores from Kruskal-Wallis.
	Returns an updated matrix.
	"""
	scores = kruskal_wallis(X, labels)
	cov_X = compute_λ_cov(X)[0]

	redundant_features = []

	for i in range(cov_X.shape[0]):
		for j in range(i + 1, cov_X.shape[1]):
			if np.abs(cov_X[i,j]) > ε:
				if scores[i] < scores[j]:
					redundant_features.append(i)
				else:
					redundant_features.append(j)

	X_new = np.delete(X, redundant_features, axis = 1)

	return X_new


def class_separation(X, labels):
	X_all = np.column_stack((X, labels))
	X_all2 = {c: X_all[X_all[:, -1] == c] for c in labels}

	return X_all, X_all2


def mdc_euclidean(X, labels, mode = "train", μ0 = None, μ1 = None):
	"""
	Euclidean minimum distance classifier
	Takes a matrix X, and respective labels
	Returns relative errors to each class
	and a total one
	Includes training and testing modes.
	"""
	def g1(x, μ1):
	    return μ1.T@x - 0.5 * (μ1.T@μ1)

	def g2(x, μ2):
	    return μ2.T@x - 0.5 * (μ2.T@μ2)


	if mode == "train":

		X_all, X_all2 = class_separation(X, labels)

		μ0 = []
		μ1 = []
		for i in range(X_all.shape[1] - 1):
			μ0.append(np.mean(X_all2[0].T[i]))
			μ1.append(np.mean(X_all2[1].T[i]))

		μ0 = np.array(μ0)
		μ1 = np.array(μ1)

		ω0_wrong = 0
		ω1_wrong = 0

		for i in range(X.shape[0]):
			if g1(X[i], μ0) > g2(X[i], μ1) and X_all[i][-1] != 0:
				ω0_wrong += 1
			elif g2(X[i], μ1) > g1(X[i], μ0) and X_all[i][-1] != 1:
				ω1_wrong += 1
		
		ε_ω0_rel = ω0_wrong/X.shape[0]
		ε_ω1_rel = ω1_wrong/X.shape[0]
		total_ε = (ω0_wrong + ω1_wrong)/X.shape[0]

		return ε_ω0_rel, ε_ω1_rel, total_ε, μ0, μ1

	elif mode == "test":
		ω0_wrong = 0
		ω1_wrong = 0

		for i in range(X.shape[0]):
			if g1(X[i], μ0) > g2(X[i], μ1) and labels[i] != 0:
				ω0_wrong += 1
			elif g2(X[i], μ1) > g1(X[i], μ0) and labels[i] != 1:
				ω1_wrong += 1
		
		ε_ω0_rel = ω0_wrong/X.shape[0]
		ε_ω1_rel = ω1_wrong/X.shape[0]
		total_ε = (ω0_wrong + ω1_wrong)/X.shape[0]

		return ε_ω0_rel, ε_ω1_rel, total_ε

def mcd_mahalanobis(X, labels, mode = "train", μ0 = None, μ1 = None,
 C_inv_avg = None):
	"""
	Mahalanobis minimum distance classifier
	Takes a matrix X, and respective labels
	Returns relative errors to each class
	and a total one.
	Includes training and testing modes.
	"""
	def g1(x, μ0, C_inv_avg): 

	    return (μ0@C_inv_avg@x - 0.5 * (μ0@C_inv_avg@μ0))

	def g2(x, μ1, C_inv_avg): 

	    return (μ1@C_inv_avg@x - 0.5 * (μ1@C_inv_avg@μ1))


	if mode == "train":
		X_all, X_all2 = class_separation(X, labels)

		data_ω0 = X_all2[0][:,:-1]
		data_ω1 = X_all2[1][:,:-1]
		C_ω0 = np.cov(data_ω0.T)
		C_ω1 = np.cov(data_ω1.T)
		C_inv_avg = np.linalg.inv((C_ω0 + C_ω1)/2)

		μ0 = []
		μ1 = []
		for i in range(X_all.shape[1] - 1):
			μ0.append(np.mean(X_all2[0].T[i]))
			μ1.append(np.mean(X_all2[1].T[i]))

		μ0 = np.array(μ0)
		μ1 = np.array(μ1)

		ω0_wrong = 0
		ω1_wrong = 0

		for i in range(X.shape[0]):
			if g1(X[i], μ0, C_inv_avg) > g2(X[i], μ1, C_inv_avg) and X_all[i][-1] != 0:
				ω0_wrong += 1
			elif g2(X[i], μ1, C_inv_avg) > g1(X[i], μ0, C_inv_avg) and X_all[i][-1] != 1:
				ω1_wrong += 1
		
		ε_ω0_rel = ω0_wrong/X.shape[0]
		ε_ω1_rel = ω1_wrong/X.shape[0]
		total_ε = (ω0_wrong + ω1_wrong)/X.shape[0]

		return ε_ω0_rel, ε_ω1_rel, total_ε, μ0, μ1, C_inv_avg

	elif mode == "test":

		ω0_wrong = 0
		ω1_wrong = 0

		for i in range(X.shape[0]):
			if g1(X[i], μ0, C_inv_avg) > g2(X[i], μ1, C_inv_avg) and labels[i] != 0:
				ω0_wrong += 1
			elif g2(X[i], μ1, C_inv_avg) > g1(X[i], μ0, C_inv_avg) and labels[i] != 1:
				ω1_wrong += 1
		
		ε_ω0_rel = ω0_wrong/X.shape[0]
		ε_ω1_rel = ω1_wrong/X.shape[0]
		total_ε = (ω0_wrong + ω1_wrong)/X.shape[0]

		return ε_ω0_rel, ε_ω1_rel, total_ε




def PCA(X, test = None):
	"""
	Projects what's initially
	X into a smaller amount of
	dimensions taking into
	account what test is chosen
	(kaiser, scree)
	"""

	λ_vals, λ_vecs = compute_λ_cov(X)[1], compute_λ_cov(X)[2]

	"""
	Get the number of
	dimensions to project,
	either with kaiser
	or scree tests.
	"""
	if test == kaiser:
		dims = kaiser(λ_vals)
	elif test == scree:
		dims = scree(λ_vals, 1)

	"""
	In descending order
	Get the indexes of the
	largest λ_vals
	"""
	λ_id = λ_vals.argsort()[::-1]
	"""
	Updates the λ_vecs given the indexes
	obtained before, and also taking into
	account the dims originated by each 
	respective test
	"""
	λ_vecs_updated = λ_vecs[:, λ_id][:, :dims]

	#Projection
	X_new = np.dot(scale_σ(X), λ_vecs_updated)


	return X_new


def run(X, labels, n_sims, classifier = None):
	"""
	Main running loop with
	a certain classifier.
	Also generates statistics
	for the total error for a
	given amount of simulations.
	"""

	#train_size = int(0.7 * X.shape[0]) #If to do without sklearn
	εs_train = []
	εs_test = []

	for _ in range(n_sims):
		X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, stratify=labels)
		"""
		Below without using sklearn.
		Although it doesn't stratify
		"""
		"""
		indices = np.random.permutation(X.shape[0])
		train_indices = indices[:train_size]
		test_indices = indices[train_size:]
		X_train = X[train_indices]
		X_test = X[test_indices]
		y_train = labels[train_indices]
		y_test = labels[test_indices]
		"""

		if classifier == "euclidean":
			μ0, μ1 = mdc_euclidean(X_train, y_train)[3:]
			εs_train.append(mdc_euclidean(X_train, y_train)[2])
			εs_test.append(mdc_euclidean(X_test, y_test, mode = "test",\
				μ0=μ0, μ1=μ1))
		elif classifier == "mahalanobis":
			μ0, μ1, C_inv_avg = mcd_mahalanobis(X_train, y_train)[3:]
			εs_train.append(mcd_mahalanobis(X_train, y_train)[2])
			εs_test.append(mcd_mahalanobis(X_test, y_test, mode = "test",\
			 μ0=μ0, μ1=μ1, C_inv_avg=C_inv_avg))
	

	print(f"Training error: {np.mean(εs_train)} ± {np.std(εs_train)}")
	print(f"Testing error: {np.mean(εs_test)} ± {np.std(εs_test)}")

	pass 

def normality_check(X, handles, α):
	"""
	Checks normality of each feature
	with the Kolmogorov-Smirnov test
	(using a significance threshold α),
	displaying also the respective
	violinplot. 
	"""

	for i in range(X.shape[1]):
		feature = X[:, i]
		p_val = kstest(feature, "norm")[1] #Gaussian dist
		
		print(f"For {handles[i]}:")
		if p_val > α:
			print("Follows gaussian dist; failed to reject H0.")
		else:
			print("Not normally distributed; rejected H0.")

		plt.violinplot(feature)
		plt.title(f"Feature {handles[i]}")
		plt.show()

	pass 				


def LDA(X, labels):
	"""
	Projects what's initially
	matrix X with whatever
	amount of features
	into a 1D space.
	Returns transformed matrix.
	"""

	X_all, X_all2 = class_separation(X, labels)

	μ0 = []
	μ1 = []
	for i in range(X_all.shape[1] - 1):
		μ0.append(np.mean(X_all2[0].T[i]))
		μ1.append(np.mean(X_all2[1].T[i]))

	μ0 = np.array(μ0)
	μ1 = np.array(μ1)

	"""
	Computing the within the class
	scatter matrix (two classes)
	"""
	S0 = np.zeros((X.shape[1], X.shape[1]))
	S1 = np.zeros((X.shape[1], X.shape[1]))
	
	for i in range(X_all2[0].shape[0]):
		S0 += np.dot((X_all2[0][i,:-1] - μ0),\
			(X_all2[0][i,:-1] - μ0).T) 
		
	for i in range(X_all2[1].shape[0]):
		S1 += np.dot((X_all2[1][i,:-1] - μ0),\
			(X_all2[1][i,:-1] - μ0).T)

	Sw = S0 + S1

	"""
	Try first with invert.
	If Sw is singular,
	run with pseudo invert.
	"""
	try:
		np.linalg.inv(Sw)
		w = np.dot(np.linalg.inv(Sw),(μ0 - μ1))
	except np.linalg.LinAlgError:
		w = np.dot(np.linalg.pinv(Sw),(μ0 - μ1))
	
	w = w.reshape(w.shape[0],1)
	X_new = np.dot(w.T, X.T)

	return X_new.T



"""
To-do list

- Kruskal-Wallis test to rank features

- Acess redundancy with cov_matrix with
threshold ε (i.e. if cov(i,j) > ε, eliminate
either i or j feature based on ranking
from Kruskal-Wallis test)

- PCA(based on either kaiser or scree tests
with n dims for projection; with the corresponding
λ_vecs)

- Also needed a plotting function

- LDA...

- Fisher LDA

- Another minimum distance classifier
"""
