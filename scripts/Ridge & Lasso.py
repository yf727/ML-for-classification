## Ridge and Lasso Regression in Python

## https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/
## http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html


import numpy as np
import pandas as pd
import random 
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 10


## 1. Why penalize the magnitude of coefficents??

## Define input array with angles from 60deg to 300deg converted to radians
x = np.array([i * np.pi/180 for i in range(60, 300, 4)])
np.random.seed(10)
y = np.sin(x) + np.random.normal(0, 0.15, len(x))
data = pd.DataFrame(np.column_stack([x, y]), columns = ['x', 'y'])
plt.plot(data['x'], data['y'], '.')

## try to estimate the sine function by polynomial regression 
## with powers of x from 1 to 15

#3 add columns for each power upto 15
for i in range(2, 16): #power of 1 is already there
	#new var will be x_power
	colname = 'x_%d'%i 
	## power 1 to 15
	data[colname] = data['x']**i 
print (data.head())

## now we want 15 different linear regression models

from sklearn.linear_model import LinearRegression
def linear_regression(data, power, models_to_plot):
	
	# initialize predictors:
	predictors = ['x']
	if power >= 2:
		predictors.extend(['x_%d'%i for i in range(2, power+1)])

	# fit the model 
	linreg = LinearRegression(normalize = True)
	linreg.fit(data[predictors], data['y'])
	y_pred = linreg.predict(data[predictors])

	# check if a plot is to be made for the entered power
	if power in models_to_plot:
		plt.subplot(models_to_plot[power])
		plt.tight_layout()
		plt.plot(data['x'], y_pred)
		plt.plot(data['x'], data['y'], '.')
		plt.title('Plot for power: %d'%power)

	# return the result in pre-defined format
	rss = sum((y_pred-data['y'])**2)
	ret = [rss]
	ret.extend([linreg.intercept_])
	ret.extend(linreg.coef_)
	return ret

## Initialize a dataframe to store the results:
col = ['rss', 'intercept'] + ['coef_x_%d'%i for i in range(1, 16)]
ind = ['model_pow_%d'%i for i in range(1, 16)]
coef_matrix_simple = pd.DataFrame(index=ind, columns=col) 

## define the powers for which a plot is required
models_to_plot = {1:231,3:232,6:233,9:234,12:235,15:236}

## iterate through all powers and assimilate results
for i in range(1,16):
	coef_matrix_simple.iloc[i-1, 0:i+2] = linear_regression(data, power = i, models_to_plot = models_to_plot)

## analysis the impact of the magnitude of coefficents

## Set the display format to be scientific for ease of analysis
pd.options.display.float_format = '{:,.2g}'.format
coef_matrix_simple


## 2. Ridge regression 

from sklearn.linear_model import Ridge

def ridge_regression(data, predictors, alpha, models_to_plot={}):

	# fit the model
	ridgereg = Ridge(alpha = alpha, normalize = True)
	ridgereg.fit(data[predictors], data['y'])
	y_pred = ridgereg.predict(data[predictors])

	# Check if a plot is to be made for the entered alpha
	if alpha in models_to_plot:
		plt.subplot(models_to_plot[alpha])
		plt.tight_layout()
		plt.plot(data['x'], y_pred)
		plt.plot(data['x'], data['y'], '.')
		plt.title('Plot for alpha: %.3g'%alpha)

	# Return the result in pre-defined format
	rss = sum((y_pred-data['y'])**2)
	ret = [rss]
	ret.extend([ridgereg.intercept_])
	ret.extend(ridgereg.coef_)
	return ret

## ridge regression for 10 models with 15 variables
predictors = ['x']
predictors.extend(['x_%d'%i for i in range(2, 16)])


alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

col = ['rss', 'intercept'] + ['coef_x_%d'%i for i in range(1, 16)]
ind = ['alpha_%.2g'%alpha_ridge[i] for i in range(0,10)]
coef_matrix_ridge = pd.DataFrame(index = ind, columns = col)

models_to_plot = {1e-15:231, 1e-10:232, 1e-4:233, 1e-3:234, 1e-2:235, 5:236}
for i in range(10):
	coef_matrix_ridge.iloc[i,] = ridge_regression(data, predictors, alpha_ridge[i], models_to_plot)

## the values of coefficients 

pd.options.display.float_format = '{:,.2g}'.format
coef_matrix_ridge

## the coefficents are not 0 
coef_matrix_ridge.apply(lambda x: sum(x.values==0), axis=1)



################### 

## template 
## https://www.analyticsvidhya.com/blog/2017/06/a-comprehensive-guide-for-linear-ridge-and-lasso-regression/

## training the model
ridgeReg = Ridge(alpha = 0.05, normalize = True)
ridgeReg.fit(x_train, y_train)
pred = ridgeReg.predict(x_cv)

## calculating mse
mse = np.mean((pred_cv - y_cv)**2)





################### 

## http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

from sklearn.linear_model import Ridge
import numpy as np
n_samples, n_features = 10, 5
np.random.seed(0)
y = np.random.randn(n_samples)
X = np.random.randn(n_samples, n_features)
clf = Ridge(alpha = 1.0)
clf.fit(X, y)





