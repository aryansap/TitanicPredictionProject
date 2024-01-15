from d_tree import DecisionTree
from random_forest import RandomForest
import numpy as np
from collections import Counter
import pickle

#training the model
from sklearn.model_selection import train_test_split
import pandas as pd
df = pd.read_csv("train.csv")
df = df.dropna()
X_traino = df
y_train = df["Survived"]
X_traino = X_traino.drop(["Survived", "Name","Ticket","PassengerId"], axis = 1)
genderonehot = pd.get_dummies(df['Sex']).astype(int)

X_traino = pd.concat([X_traino,genderonehot],axis = 1)
X_traino = X_traino.drop(["Embarked","Cabin","Sex"], axis = 1)
X_train = X_traino.values
y_train = y_train.values
clf = RandomForest()
clf.fit(X_train,y_train)

# Saving model to disk
pickle.dump(clf, open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
