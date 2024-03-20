from utils import *

features, labels, handles = initial_processing("default_of_credit_card_clients.xls")

classifiers = ["euclidean", "mahalanobis"]

for classifier in classifiers:
	for i in np.linspace(0.85, 0.95, 10):
		comm = f"run((redundancy_check(features, {i}, labels)),\
			labels, 10, 0.7, 0.15,\
			classifier = '{classifier}')"

		print(f"Command: {comm}")

		exec(comm)