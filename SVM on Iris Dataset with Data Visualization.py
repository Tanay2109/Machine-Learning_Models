import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

data=pd.read_csv('/content/Iris.csv')

data.info()

features=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
Prediction=['Species']

X=data[features].values
Y=data[Prediction].values

plt.plot(X)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.40,random_state=42)

from sklearn import svm

clf=svm.SVC(kernel='linear')

clf.fit(X_train,Y_train)

Y_predict=clf.predict(X_test)

from sklearn import metrics

print("Accuracy score:", metrics.accuracy_score(Y_test,Y_predict)*100)

