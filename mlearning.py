# Data types
   #numerical
   #categorical
   #ordinal

#mean mode and median
#use numpy to get mean 
import numpy
speed=[99,86,87,88,111,86,103,87,94,78,77,85,86]
x=numpy.mean(speed)
print(x)

#use numpy to get median 
import numpy
speed=[99,86,87,88,111,86,103,87,94,78,77,85,86]
x=numpy.median(speed)
print(x)

#use scipy for mode()
from scipy import stats
speed=[99,86,87,88,111,86,103,87,94,78,77,85,86]
x=stats.mode(speed)
print(x)

#standard deviation numpy std()
import numpy
speed=[86,87,88,86,87,85,86]
x=numpy.std(speed)
print(x)

#variance numpy var()
import numpy
speed=[32,111,138,28,59,77,97]
x=numpy.var(speed)
print(x)

#percentiles
import numpy
ages=[5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]
x=numpy.percentile(ages,75)
print(x)

import numpy
ages=[5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]
x=numpy.percentile(ages,90)
print(x)

#data distribution 
#create an array
import numpy
x=numpy.random.uniform(0.0,5.0,250)
print(x)

#visualize the data set using a histogram
#import matplotlib.pyplot as plt
#import numpy
#x=numpy.random.uniform(0.0,5.0,250)
#plt.hist(x,5)
#plt.show()

#big data distribution
import numpy 
x=numpy.random.uniform(0.0,5.0,10000)
print(x)

#visualize the data set
#import numpy
#import matplotlib.pyplot as plt
#x=numpy.random.uniform(0.0,5.0,10000)
#plt.hist(x,100)
#plt.show()

#normal data distribution(the bell curve)
#import numpy
#import matplotlib.pyplot as plt
#x=numpy.random.normal(5.0,1.0,100000)
#plt.hist(x,100)
#plt.show()
#5.0 is the mean value
#1.0 is the standard deviation

#scatter plot
#import matplotlib.pyplot as plt
#x=[5,7,8,7,2,17,2,9,4,11,12,9,6]
#y=[99,86,87,88,111,86,103,87,94,78,77,85,86]
#plt.scatter(x,y)
#plt.show()

#random data distributions
#import numpy
#import matplotlib.pyplot as plt
#x=numpy.random.normal(5.0,1.0,1000)
#y=numpy.random.normal(10.0,2.0,1000)
#plt.scatter(x,y)
#plt.show()

#linear regression
#start by drawing a scatter plot 
#import matplotlib.pyplot as plt
#x=[5,7,8,7,2,17,2,9,4,11,12,9,6]
#y=[99,86,87,88,111,86,103,87,94,78,77,85,86]
#plt.scatter(x,y)
#plt.show()

#then import scipy to draw a line of linear regression
'''
import matplotlib.pyplot as plt
from scipy import stats
x=[5,7,8,7,2,17,2,9,4,11,12,9,6]
y=[99,86,87,88,111,86,103,87,94,78,77,85,86]
slope,intercept,r,p,std_err=stats.linregress(x,y)
def myfunc(x):
   return slope*x+intercept
mymodel=list(map(myfunc,x))
plt.scattter (x,y)
plt.plot(x,mymodel)
plt.show() 
'''  

#R for relationship
from scipy import stats
x=[5,7,8,7,2,17,2,9,4,11,12,9,6]
y=[99,86,87,88,111,86,103,87,94,78,77,85,86]
slope,intercept,r,p,std_err=stats.linregress(x,y)
print(r)

#predict future values 
#predict the speed for a 10 year old car
from scipy import stats
x=[5,7,8,7,2,17,2,9,4,11,12,9,6]
y=[99,86,87,88,111,86,103,87,94,78,77,85,86]
slope,intercept,r,p,std_err=stats.linregress(x,y)
def myfunc(x):
   return slope*x+intercept
speed=myfunc(10)
print(speed)   

#bad fit
'''
import matplotlib.pyplot as plt
from scipy import stats
x=[89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
y=[21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]
slope,intercept,r,p,std_err=stats.linregress(x,y)
def myfunc(x):
   return slope*x+intercept
mymodel=list(map(myfunc,x))
plt.scatter(x,y)
plt.plot(x,mymodel)
plt.show()   
'''
#find the relationship (r)
import numpy
from scipy import stats
x=[89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
y=[21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]
slope,intercept,r,p,std_err=stats.linregress(x,y)
print(r)

#polynomial regression
'''
import matplotlib.pyplot as plt
x=[1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y=[100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
plt.scatter(x,y)
plt.show()
'''
#import numpy to draw a line of polynomial regression
'''
import numpy
import matplotlib.pyplot as plt
x=[1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y=[100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
mymodel=numpy.poly1d(numpy.polyfit(x,y,3))
myline=numpy.linespace(1,22,100)
plt.scatter(x,y)
plt.plot(myline,mymodel(myline))
plt.show()
'''

#R-squared
import numpy
from sklearn.metrics import r2_score
x=[1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y=[100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
mymodel=numpy.poly1d(numpy.polyfit(x,y,3))
print(r2_score(y,mymodel(x)))

#predict future values
import numpy
from sklearn.metrics import r2_score
x=[1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y=[100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
mymodel=numpy.poly1d(numpy.polyfit(x,y,3))
speed=mymodel(17)
print(speed)

#bad fit
'''
import numpy
import matplotlib.pyplot as plt
x=[89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
y=[21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]
mymodel=numpy.poly1d(numpy.polyfit(x,y,3))
myline=numpy.linspace(2,95,100)
plt.scatter(x,y)
plt.plot(myline,mymodel(myline))
plt.show()
'''

#the r-squared value of the above is very low
import numpy
from sklearn.metrics import r2_score
x=[89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
y=[21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]
mymodel=numpy.poly1d(numpy.polyfit(x,y,3))
print(r2_score(y,mymodel(x)))

'''
import pandas
from sklearn import linear_model
df=pandas.read_csv("data.csv")
X=df[['weight','volume']]
y=df['CO2']
regr=linear_model.linearRegression()
regr.fit(X,y)
predictedCO2=regr.predict([[2300,1300]])
print(predictedCO2)
'''
#coefficient
'''
import pandas
from sklearn import linear_model

df = pandas.read_csv("data.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, y)

print(regr.coef_)
'''

#scale features
#we use standardizing method
#formula z=(x-u)/s
#z=new value
#x=original value 
#u=mean
#s=standard deviation
'''
import pandas 
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
df=pandas.read_csv("data.csv")
X=df[['weight','volume']]
scaledX=scale.fit_transform(X)
print(scaledX)
'''
#Train/Test
#our data set ilustrates 100 customers in a shop , and their shopping habits.
'''
import numpy
import matplotlib.pyplot as plt 
numpy.random.seed(2)
x=numpy.random.normal(3,1,100)
y=numpy.random.normal(150,40,100)/x
plt.scatter(x,y)
plt.show()
'''

#we train and test the data set above 
#we train 80% and test 20%
'''
import numpy
import matplotlib.pyplot as plt 
numpy.random.seed(2)
x=numpy.random.normal(3,1,100)
y=numpy.random.normal(150,40,100)/x
train_x=x[:80]
train_y=y[:80]

test_x=x[80:]
test_y=y[80:]

plt.scatter(train_x,train_y)
plt.show()
'''

#display the testing set
'''
import numpy
import matplotlib.pyplot as plt 
numpy.random.seed(2)
x=numpy.random.normal(3,1,100)
y=numpy.random.normal(150,40,100)/x
train_x=x[:80]
train_y=y[:80]

test_x=x[80:]
test_y=y[80:]

plt.scatter(test_x,test_y)
plt.show()
'''

#fit the data set
#draw a polynomial regression line through the data points
'''
import numpy
import matplotlib.pyplot as plt 
numpy.random.seed(2)

x=numpy.random.normal(3,1,100)
y=numpy.random.normal(150,40,100)/x

train_x=x[:80]
train_y=x[:80]

test_x=x[80:]
test_y=y[80:]

mymodel=numpy.poly1d(numpy.polyfit(train_x,train_y,4))
myline=numpy.linspace(0,6,100)

plt.scatter(train_x,train_y)
plt.plot(myline,mymodel(myline))
plt.show()
'''

#we use the R-squared to get the relationship of our model
import numpy
from sklearn.metrics import r2_score 
numpy.random.seed(2)
x=numpy.random.normal(3,1,100)
y=numpy.random.normal(150,40,100)/x 
train_x=x[:80]
train_y=y[:80]
test_x=x[80:]
test_y=y[80:]
mymodel=numpy.poly1d(numpy.polyfit(train_x,train_y,4))
r2=r2_score(train_y,mymodel(train_x))
print(r2)

#predict values 
import numpy
from sklearn.metrics import r2_score 
numpy.random.seed(2)
x=numpy.random.normal(3,1,100)
y=numpy.random.normal(150,40,100)/x 
train_x=x[:80]
train_y=y[:80]
test_x=x[80:]
test_y=y[80:]
mymodel=numpy.poly1d(numpy.polyfit(train_x,train_y,4))

print(mymodel(5))

#machine learning decision tree
import pandas as pd
data = {
    'Feature1': [5, 10, 15, 20, 25, 30, 35, 40],
    'Feature2': [1, 2, 3, 4, 5, 6, 7, 8],
    'Label': ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B']
}

df = pd.DataFrame(data)
print(df)

#data csv
import pandas as pd
data={
   'Age':[36,42,23,52,43,44,66,35,52,35,24,18,45],
   'Experience':[10,12,4,4,21,14,3,14,13,5,3,3,9],
   'Rank':[9,4,6,4,8,5,7,9,7,9,5,7,9],
   'Nationality':["UK","USA","N","USA","USA","UK","N","UK","N","N","USA","UK","UK"],
   'Go':["NO","NO","NO","NO","YES","NO","YES","YES","YES","YES","NO","YES","YES"]
}
df=pd.DataFrame(data)
print(df)

#All data in the tree should be numerical thus we change using the map function()

d={"UK":0,"USA":1,"N":2}
df['Nationality']=df['Nationality'].map(d)
d={'YES':1,"NO":0}
df['Go']=df['Go'].map(d)

print(df)

#let x be the feature columns and y be the target columns
features=['Age','Experience','Rank','Nationality']
x=df[features]
y=df['Go']
print(x)
print(y)

#how to save a data set as an csv 
import pandas as pd 
data={
   'Age':[36,42,23,52,43,44,66,35,52,35,24,18,45],
   'Experience':[10,12,4,4,21,14,3,14,13,5,3,3,9],
   'Rank':[9,4,6,4,8,5,7,9,7,9,5,7,9],
   'Nationality':["UK","USA","N","USA","USA","UK","N","UK","N","N","USA","UK","UK"],
   'Go':["NO","NO","NO","NO","YES","NO","YES","YES","YES","YES","NO","YES","YES"]
}
df=pd.DataFrame(data)
df.to_csv('decision_data.csv',index=False)
print("CSV file saved successfully.")

#create and display a decision tree
'''
import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
df=pandas.read_csv(data)
d={'UK':0,'USA':1,'N':2}
df['Nationality']=df['Nationality'].map(d)
d={'YES':0,'NO':1}
df['Go']=df['Go'].map(d)
features=['Age','Expirience','Rank','Nationality']
x=df[features]
y=df['Go']
dtree=DecisionTreeClassifier()
dtree=dtree.fit(x,y)
tree.plot_tree(dtree,feature_names=features)
'''