import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import time #For timing purposes
from sklearn.metrics import roc_curve, auc
#from sklearn.model_selection import train_test_split #If want to use sklearn
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

	"""
	Need to order λ_list (descending)
	"""
	λ_list_sort = np.sort(λ_list)[::-1]

	PC = np.sum(λ_list_sort >= 1)

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
	"""
	Need to order λ_list (descending)
	"""
	λ_list_sort = np.sort(λ_list)[::-1]

	"""
	Compute diferences between
	consecutive eigenvals
	"""
	Δ = np.abs(np.diff(λ_list_sort))
	
	#Find the idx where Δ < ε
	idx = np.argmax(Δ < ε)

	PC = idx + 1 

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


def kept_σ(λ_list, n):
	"""
	Takes a list of eigenvalues
	Computes the kept variance
	for n amount of features
	"""

	"""
	Need to order λ_list (descending)
	"""
	λ_list_sort = np.sort(λ_list)[::-1]

	# For the first n features
	kept_σ = np.sum(λ_list_sort[:n]**2)

	total_σ = np.sum(λ_list_sort**2)

	return kept_σ/total_σ


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
	"""
	Returns X_all with all features
	and labels. And also returns X_all2
	dictionary separated by classes.
	"""

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
	def g(x, μ):
		return μ.T@x - 0.5 * (μ.T@μ)

	X_all, X_all2 = class_separation(X, labels)


	if mode == "train":

		#Without last col
		μ0 = np.mean(X_all2[0].T[:-1], axis = 1)
		μ1 = np.mean(X_all2[1].T[:-1], axis = 1)


		"""
		Gives an array with boolean values
		and sums True instances.
		If the condition between ()
		is met, we have a True instance
		for that sample.
		"""
		ω0_wrong = np.sum((g(X.T, μ0) > g(X.T, μ1)) & (labels != 0))
		ω1_wrong = np.sum((g(X.T, μ1) > g(X.T, μ0)) & (labels != 1))
		
		ε_ω0_rel = ω0_wrong/X.shape[0]
		ε_ω1_rel = ω1_wrong/X.shape[0]
		total_ε = (ω0_wrong + ω1_wrong)/X.shape[0]

		return ε_ω0_rel, ε_ω1_rel, total_ε, μ0, μ1

	elif mode == "val":
		

		ω0_wrong = np.sum((g(X.T, μ0) > g(X.T, μ1)) & (labels != 0))
		ω1_wrong = np.sum((g(X.T, μ1) > g(X.T, μ0)) & (labels != 1))
		
		
		ε_ω0_rel = ω0_wrong/X.shape[0]
		ε_ω1_rel = ω1_wrong/X.shape[0]
		total_ε = (ω0_wrong + ω1_wrong)/X.shape[0]

		return ε_ω0_rel, ε_ω1_rel, total_ε	

	elif mode == "test":

		ω0_wrong = np.sum((g(X.T, μ0) > g(X.T, μ1)) & (labels != 0))
		ω1_wrong = np.sum((g(X.T, μ1) > g(X.T, μ0)) & (labels != 1))
		

		"""
		To get ROC and compute AUC
		Vectorized form;
		Need to transpose X, in order
		to have each sample representing
		a col.
		"""
		ω_scores = g(X.T, μ1) - g(X.T, μ0)

		ε_ω0_rel = ω0_wrong/X.shape[0]
		ε_ω1_rel = ω1_wrong/X.shape[0]
		total_ε = (ω0_wrong + ω1_wrong)/X.shape[0]

		C = np.array([[X_all2[1].shape[0],ω1_wrong],\
		[ω0_wrong,X_all2[0].shape[0]]]) #Confusion matrix

		return ε_ω0_rel, ε_ω1_rel, total_ε, C, ω_scores

def mcd_mahalanobis(X, labels, mode = "train", μ0 = None, μ1 = None,
 C_inv_avg = None):
	"""
	Mahalanobis minimum distance classifier
	Takes a matrix X, and respective labels
	Returns relative errors to each class
	and a total one.
	Includes training and testing modes.
	"""
	def g(x, μ, C_inv_avg): 

	    return (μ@C_inv_avg@x - 0.5 * (μ@C_inv_avg@μ))

	X_all, X_all2 = class_separation(X, labels)

	if mode == "train":
		data_ω0 = X_all2[0][:,:-1]
		data_ω1 = X_all2[1][:,:-1]
		C_ω0 = np.cov(data_ω0.T)
		C_ω1 = np.cov(data_ω1.T)
		C_inv_avg = np.linalg.inv((C_ω0 + C_ω1)/2)

		μ0 = np.mean(X_all2[0].T[:-1], axis = 1)
		μ1 = np.mean(X_all2[1].T[:-1], axis = 1)

		ω0_wrong = np.sum((g(X.T, μ0, C_inv_avg) > g(X.T, μ1, C_inv_avg)) &\
		 (labels != 0))
		ω1_wrong = np.sum((g(X.T, μ1, C_inv_avg) > g(X.T, μ0, C_inv_avg)) &\
		 (labels != 1))
		
		
		ε_ω0_rel = ω0_wrong/X.shape[0]
		ε_ω1_rel = ω1_wrong/X.shape[0]
		total_ε = (ω0_wrong + ω1_wrong)/X.shape[0]

		return ε_ω0_rel, ε_ω1_rel, total_ε, μ0, μ1, C_inv_avg

	elif mode == "val":

		ω0_wrong = np.sum((g(X.T, μ0, C_inv_avg) > g(X.T, μ1, C_inv_avg)) &\
		 (labels != 0))
		ω1_wrong = np.sum((g(X.T, μ1, C_inv_avg) > g(X.T, μ0, C_inv_avg)) &\
		 (labels != 1))
		
		ε_ω0_rel = ω0_wrong/X.shape[0]
		ε_ω1_rel = ω1_wrong/X.shape[0]
		total_ε = (ω0_wrong + ω1_wrong)/X.shape[0]

		return ε_ω0_rel, ε_ω1_rel, total_ε



	elif mode == "test":

		ω0_wrong = np.sum((g(X.T, μ0, C_inv_avg) > g(X.T, μ1, C_inv_avg)) &\
		 (labels != 0))
		ω1_wrong = np.sum((g(X.T, μ1, C_inv_avg) > g(X.T, μ0, C_inv_avg)) &\
		 (labels != 1))

		"""
		To get ROC and compute AUC
		"""
		ω_scores = g(X.T, μ1, C_inv_avg) - g(X.T, μ0, C_inv_avg)
		
		ε_ω0_rel = ω0_wrong/X.shape[0]
		ε_ω1_rel = ω1_wrong/X.shape[0]
		total_ε = (ω0_wrong + ω1_wrong)/X.shape[0]

		C = np.array([[X_all2[1].shape[0],ω1_wrong],\
		[ω0_wrong,X_all2[0].shape[0]]]) #Confusion matrix

		return ε_ω0_rel, ε_ω1_rel, total_ε, C, ω_scores




def PCA(X, test = None):
	"""
	Projects what's initially
	X into a smaller amount of
	dimensions taking into
	account what test is chosen
	(kaiser, scree)
	"""

	λ_vals, λ_vecs = compute_λ_cov(X)[1:]

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
	largest λ_vals, and also
	taking into account the
	dims originated by each 
	respective test.
	"""
	λ_id = np.argsort(λ_vals)[::-1][:dims]

	λ_vecs_updated = λ_vecs[:, λ_id]

	#Projection
	X_new = np.dot(scale_σ(X), λ_vecs_updated)


	return X_new


def run(X, labels, n_sims, train_ω = None,
 val_ω = None, classifier = None, testing = False,
 plot = False):
	"""
	Main running loop with
	a certain classifier.
	Also generates statistics
	for the total error for a
	given amount of simulations.
	"""
	τ_i = time.time()

	train_size = int(train_ω * X.shape[0]) #If to do without sklearn
	val_size = int(val_ω * train_size)
	εs_train = []
	εs_val = []



	"""
	Want this before the loop
	accounting for multiple 
	simulations.
	Don't want my testing set changing 
	from simulation to simulation.
	"""
	indices = np.random.permutation(X.shape[0])
	test_indices = indices[train_size:]
	X_test = X[test_indices]
	y_test = labels[test_indices]
	for _ in range(n_sims):
		#X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, stratify=labels)
		"""
		Below without using sklearn.
		Although it doesn't stratify
		"""
		"""
		Not doing k-fold CV.
		Doing random subsampling.
		Don't get the assurance
		that every sample eventually
		gets into a validation set.
		"""
		train_val_indices = np.random.permutation(indices[:train_size])
		val_indices = train_val_indices[:val_size]
		train_indices = train_val_indices[val_size:]
		X_train = X[train_indices]
		X_val = X[val_indices]
		y_train = labels[train_indices]
		y_val = labels[val_indices]
		


		if classifier == "euclidean":
			ε, μ0, μ1 = mdc_euclidean(X_train, y_train)[2:]
			εs_train.append(ε)
			εs_val.append(mdc_euclidean(X_val, y_val, mode = "val",\
				μ0=μ0, μ1=μ1)[2])

		
		elif classifier == "mahalanobis":
			ε, μ0, μ1, C_inv_avg = mcd_mahalanobis(X_train, y_train)[2:]
			εs_train.append(ε)
			εs_val.append(mcd_mahalanobis(X_val, y_val, mode = "val",\
			 μ0=μ0, μ1=μ1, C_inv_avg=C_inv_avg)[2])
	

	print(f"Training error: {np.mean(εs_train):.4f} ± {np.std(εs_train):.4f}")
	print(f"Validation error: {np.mean(εs_val):.4f} ± {np.std(εs_val):.4f}")

	if testing == True:
		if classifier == "euclidean":
			#Returning global error, and confusion matrix
			ε_test, C, ω_scores = mdc_euclidean(X_test, y_test, mode = "test",\
				μ0=μ0, μ1=μ1)[2:]
			

		elif classifier == "mahalanobis":
			#Returning global error, and confusion matrix
			ε_test, C, ω_scores = mcd_mahalanobis(X_test, y_test, mode = "test",\
			 μ0=μ0, μ1=μ1, C_inv_avg=C_inv_avg)[2:]
			

		print(f"Testing error: {ε_test:.4f}")
		"""
		In the form
		[TP FN
		 FP TN]
		"""
		print(f"Confusion matrix: {C}")
		SS = C[0,0]/(C[0,0] + C[0,1]) #Sensitivity
		SP = C[1,1]/(C[1,1] + C[1,0]) #Specificity
		print(f"Sensitivity: {SS:.4f}")
		print(f"Specificity: {SP:.4f}")

		fpr, tpr, thresholds = roc_curve(y_test, ω_scores)
		AUC = auc(fpr, tpr)
		print(f"AUC: {AUC:.4f}")

		if plot == True:
			plt.plot(fpr, tpr, "k",
				label = f"ROC (AUC = {AUC:.4f})")
			plt.plot([0,1],[0,1], "r--",
				label = "Chance line")
			plt.xlabel("FPR")
			plt.ylabel("TPR")
			plt.legend()


	τ_f = time.time()
	Δτ = τ_f - τ_i
	print(f"Execution time: {Δτ:.2f}")
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

	μ0 = np.mean(X_all2[0].T[:-1], axis = 1)
	μ1 = np.mean(X_all2[1].T[:-1], axis = 1)


	"""
	Computing the within the class
	scatter matrix (two classes)
	"""

	"""
	If χ is the number of features,
	both S0 and S1 should have
	shapes (χ, χ).
	In order for this to happen,
	the first argument of np.dot()
	is the one needing transposed.
	Otherwise, we have S0 and S1 with
	shapes (n, n), n being the number
	of samples associated to each class.
	"""

	S0 = np.dot((X_all2[0][:, :-1] - μ0).T,\
	 (X_all2[0][:, :-1] - μ0))
	S1 = np.dot((X_all2[1][:, :-1] - μ1).T,\
	 (X_all2[1][:, :-1] - μ1))


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
