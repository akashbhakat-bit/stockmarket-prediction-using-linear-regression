import pandas as pd
import quandl
import math
import numpy as np
from sklearn.model_selection import cross_validate, train_test_split
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression

#cross validation for training and testing, svm can be used to do regression, preprossecing scale karta hai data ko


quandl.ApiConfig.api_key='M578LbTBRq-Y7ayJ9z-s'
df=quandl.get('WIKI/GOOGL')
#print(df.head())
df=df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT']=(df['Adj. High'] - df['Adj. Close'])/df['Adj. Close']*100
df['PCT_change']=(df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open']*100
df=df[['Adj. Close','HL_PCT','PCT_change', 'Adj. Volume']]
#print(df.head())
#forecast_col can be used anywhere. agar naya prediction karna hai other than stock jo yaha pe ho raha hai, toh iss comment ke uppar ko value/lines mei change hoga
#print(len(df))
forecast_col= 'Adj. Close'
df.fillna(-99999, inplace=True)
#agar nan data hai toh usse kaam nahi kar sakte ho. toh usko replcae karna hoga. sacrifice data nahi karna hai
forecast_out=int(math.ceil(0.1*len(df)))
#print(forecast_out)
#math ceil will round it upn to 0.1 it will return a float it will try to predict out 10% of the data frame.
#it will predict 10 days now. agar 0.1 ko 0.01 kar diya toh kal ka predict krega
df['label']=df[forecast_col].shift(-forecast_out)
#humne run kiya toh dekha ki .1 bahut bada value de raha hai. isliye usko .01 kiya
#print(df.head())

X=np.array(df.drop(['label'],1))#drop leabal column
X=preprocessing.scale(X)

X=X[:-forecast_out]
X_lately=X[-forecast_out:]
df.dropna(inplace=True)

Y= np.array(df['label'])

#X=X[:-forecast_out+1]
#it ensures only values that has value for Y
#lekin hamne label drop kar diya toh uska ab zaroorat nahi hai
y=np.array(df['label'])
#creating trainig and testing set
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

#clf=svm.SVR()
clf=LinearRegression()#now one job is per tiem. we can change it by n_jobs=10, n_jobs=-1 will give the fastest speed
clf.fit(X_train,Y_train)
accuracy=clf.score(X_test,Y_test)
#if trained testing is done on another thing. Accuracy can give 2 values also. not here
#print(accuracy)
#now let us use svm simple vector regression

#now predicitn on the basis of X data
forecast_set=clf.predict(X_lately)#here can be array or one value prediction perr value
print(forecast_set,accuracy,forecast_out)




