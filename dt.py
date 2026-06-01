import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
df=load_iris()
x = pd.DataFrame(df['data'])
y = pd.DataFrame(df['target'])
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0000)
print(x_test.shape)
print(y_test.shape)
from sklearn.tree import DecisionTreeClassifier
d = DecisionTreeClassifier()
d.fit(x_train,y_train)
y_pred = d.predict(x_test)
from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(d,filled=True)
plt.show()
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(accuracy_score(y_pred,y_test))
print(classification_report(y_pred,y_test))
print(confusion_matrix(y_pred,y_test))
