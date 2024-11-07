import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_set=pd.read_csv("Salary_Data.csv")
X=data_set.iloc[:,:-1].values
y=data_set.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(X_train,y_train)

plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regression.predict(X_train),color='blue')
plt.title("Linear Model (Training values)")
plt.xlabel("Years of Experience")
plt.ylabel("Salery")


plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regression.predict(X_train),color='blue')
plt.title("Linear Model (Testing values)")
plt.xlabel("Years of Experience")
plt.ylabel("Salery")

plt.show()

