
#importing libraries

import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#importing the dataset

url= "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names=['sepal-length','sepal-width','petal-length','petal-width','class']
dataset=pd.read_csv(url,names=names)

#data visualisation

print(dataset.shape)
print(dataset.head(10))


dataset.plot(kind='box', subplots=True, layout= (2,2), sharex=False, sharey=False)
plt.show()

dataset.hist()
plt.show()

scatter_matrix(dataset)
plt.show()


#data munging

array=dataset.values
X=array[:,0:4]
y=array[:,4]
validation_size=0.20
seed=7
X_train,X_test, y_train, y_test= model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)


seed=7
scoring='accuracy'


#putting out different models and then choosing the best one


models=[]
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))



results=[]
names=[]

kfold = model_selection.KFold(n_splits=10, random_state=seed)

for name, model in models:
    cv_results=model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(name)
    msg="%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)



	
#we find svm to be best one    
    
svm=SVC()
svm.fit(X_train, y_train)
predictions=svm.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
