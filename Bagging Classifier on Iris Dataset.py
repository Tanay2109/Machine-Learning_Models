import numpy as np

import pandas as pd

data=pd.read_csv('/content/Iris.csv')

features=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']

Prediction=['Species']

X=data[features].values
Y=data[Prediction].values

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)

from sklearn.ensemble import BaggingClassifier

from sklearn.tree import DecisionTreeClassifier

bg=BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1,n_estimators=20)

bg.fit(X_train,Y_train)

accuracy=bg.score(X_test,Y_test)

from sklearn import metrics

print("Accuracy Score;"+str(accuracy*100))

