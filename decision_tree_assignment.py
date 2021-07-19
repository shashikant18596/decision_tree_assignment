import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn
from pandas import Series,DataFrame
from pylab import rcParams
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report

df = pd.read_csv("https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv")
pd.set_option("display.max_columns",None)
df.fillna(df['Age'].mean(),inplace=True)
#print(df.describe())
#print(f"df \n {df}")
df1 = df[['Pclass','Sex', 'Age', 'SibSp','Parch','Fare','Survived']]
#print(f"df1 \n {df1}")
#print(f"describe \n {df.describe()}")
x = df1.drop(columns=df1[['Survived']],axis=1)
#print(f"attribute column \n {x}")
y= df1.Survived
#print(f"result column \n {y}")
dummies = pd.get_dummies(x.Sex)
concat = pd.concat([x,dummies],axis=1).drop(columns=x[['Sex']],axis=1)
print(concat)
x_train,x_test,y_train,y_test =train_test_split(concat,y,train_size=0.4)
model=DecisionTreeClassifier()
model.fit(x_train,y_train)
#print(model.predict(x_test))
print(model.score(x_test,y_test))


