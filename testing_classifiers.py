from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
from numpy import array,reshape
from pickle import dump,load
from helper_functions import *

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


window_size = 50
limit = 10 ** 3*3

training_dataset = parse_dataset('Raw Mat Files (CLA)/CLASubjectB1510193StLRHand.mat',window_size,limit)

X = reshape(load(open('same_patient_encoded.pkl','rb')),[3000,2])
y = array(training_dataset[1])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

for clf in classifiers:
    clf.fit(X_train,y_train)
    training_acc = clf.score(X_train,y_train)
    testing_acc = accuracy_score(y_test,clf.predict(X_test))

    print(f'{clf}: Training accuracy: {training_acc}  Testing accuracy: {testing_acc}')










