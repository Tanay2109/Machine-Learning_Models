import numpy as np

import pandas as pd

data=pd.read_csv('/content/diabetes.csv')

data.info()

features=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
Predictions=["Outcome"]

X=data[features].values
Y=data[Predictions].values

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)

from sklearn.linear_model import LogisticRegression

clf=LogisticRegression()

clf.fit(X_train,Y_train)

Y_predict=clf.predict(X_test)

from sklearn import metrics

print("Accuracy Score;",metrics.accuracy_score(Y_test,Y_predict)*100)

