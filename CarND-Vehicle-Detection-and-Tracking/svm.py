import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def train_svm(X_tr, y_tr, X_val=None, y_val=None):
	# standardize features with sklearn preprocessing
	#Compute the mean and std to be used for later scaling
	feature_scaler_tr = StandardScaler().fit(X_tr)  # per-column scaler
	#Perform standardization by centering and scaling
	X_tr_scaled = feature_scaler_tr.transform(X_tr)

	clf = LinearSVC(C=1.0)
	clf.fit(X_tr_scaled, y_tr)

	if(X_val and y_val):
		#feature_scaler_val = StandardScaler().fit(X_val)
		X_val_scaled = feature_scaler.transform(X_val)
		acc = accuracy_score(y_val, clf.predict(X_val_scaled))
		print('Validation accuracy: ', round(acc, 4))
    
	with open('data/svm_trained.pickle', 'wb') as f:
		pickle.dump(clf, f)
	with open('data/feature_scaler.pickle', 'wb') as f:
		pickle.dump(feature_scaler_tr, f)

	return clf, feature_scaler_tr

def load_model():
	# load pretrained svm classifier
	clf = pickle.load(open('data/svm_trained.pickle', 'rb'))

	# load feature scaler fitted on training data
	feature_scaler = pickle.load(open('data/feature_scaler.pickle', 'rb'))

	return clf, feature_scaler