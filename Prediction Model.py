import pandas as pd
import numpy as np 
import matplotlib.pylab as plt 
import seaborn as seaboraninstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
from statistics import mean
import pickle

dataset = pd.read_csv('Egypt (COVID-19).csv')

dataset.dropna(inplace=True)

x = dataset.iloc[:,2].values.reshape(-1, 1) 
y = dataset.iloc[:,4].values.reshape(-1, 1) 

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2 , random_state = 2)
    
model = LinearRegression()

model.fit(xtrain,ytrain)

yp = model.predict(xtest)

print ('Accuracy of model:',r2_score(ytest , yp)*100,"%")
print ('Mean square error:' , metrics.mean_squared_error(ytest , yp))
print ('Root mean square error:' , np.sqrt(metrics.mean_squared_error(ytest , yp)))


pickle.dump(model, open('Pickle Module.pkl','wb'))

