#confusion matrix
'''
import matplotlib.pyplot as plt
import numpy
from sklearn import metrics

actual=numpy.random.binomial(1,.9,size=1000)
predicted=numpy.random.binomial(1,.9,size=1000)

confusion_matrix=metrics.confusion_matrix(actual,predicted)

cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,display_labels=[0,1])

cm_display.plot()
plt.show()
'''
#accuracy
#(True Positive + True Negative)/Total predictions
import numpy
from sklearn import metrics

actual=numpy.random.binomial(1,.9,size=1000)
predicted=numpy.random.binomial(1,.9,size=1000)

Accuracy=metrics.accuracy_score(actual,predicted)
print(Accuracy)

#Precision
#True positive/(True positive + false positive)
import numpy
from sklearn import metrics 

actual=numpy.random.binomial(1,.9,size=1000)
predicted=numpy.random.binomial(1,.9,size=1000)

Precision=metrics.precision_score(actual,predicted)
print(Precision)

#Sensitivity(recall)
#True positive/(True positive + False negative)
import numpy 
from sklearn import metrics 

actual=numpy.random.binomial(1,.9,size=1000)
predicted=numpy.random.binomial(1,.9,size=1000)

Sensitivity_recall=metrics.recall_score(actual,predicted)
print(Sensitivity_recall)

#Specificity
#True Negative/(True Negative + False positive)
import numpy 
from sklearn import metrics 

actual=numpy.random.binomial(1,.9,size=1000)
predicted=numpy.random.binomial(1,.9,size=1000)

Specificity=metrics.recall_score(actual,predicted,pos_label=0)
print(Specificity)

#F-score(harmonic mean of precision and sensitivity)
#2*((precision*sensitivity)/(precision + sensitivity))
import numpy 
from sklearn import metrics 

actual=numpy.random.binomial(1,.9,size=1000)
predicted=numpy.random.binomial(1,.9,size=1000)

F1_score=metrics.f1_score(actual,predicted)
print(F1_score)

#all calculations
import numpy
from sklearn import metrics

actual = numpy.random.binomial(1,.9,size = 1000)
predicted = numpy.random.binomial(1,.9,size = 1000)

Accuracy = metrics.accuracy_score(actual, predicted)
Precision = metrics.precision_score(actual, predicted)
Sensitivity_recall = metrics.recall_score(actual, predicted)
Specificity = metrics.recall_score(actual, predicted, pos_label=0)
F1_score = metrics.f1_score(actual, predicted)

#metrics:
print({"Accuracy":Accuracy,"Precision":Precision,"Sensitivity_recall":Sensitivity_recall,"Specificity":Specificity,"F1_score":F1_score})

#hierarchical clustering
'''
import numpy as np 
import matplotlib.pyplot as plt

x=[4,5,10,4,3,11,14,6,10,12]
y=[21,19,24,17,16,25,24,22,21,21]

plt.scatter(x,y)
plt.show()
'''

#now we compute the ward linkage using euclidian distance, and visualize it using a dendrogram:
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

x=[4,5,10,4,3,11,14,6,10,12]
y=[21,19,24,17,16,25,24,22,21,21]

data=list(zip(x,y))
linkage_data=linkage(data,method='ward',metric='euclidean')
dendrogram(linkage_data)

plt.show()
'''

#we use sklearn the visualize on a 2-dimensional plot
'''
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import AgglomerativeClustering

x=[4,5,10,4,3,11,14,6,10,12]
y=[21,19,24,17,16,25,24,22,21,21]

data=list(zip(x,y))

hierarchical_cluster=AgglomerativeClustering(n_clusters=2,metric='euclidean',linkage='ward')
labels=hierarchical_cluster.fit_predict(data)

plt.scatter(x,y,c=labels)
plt.show()
'''

#logistic regression
import numpy
from sklearn import linear_model 
x=numpy.array([3.78,2.44,2.09,0.14,1.72,1.65,4.92,4.37,4.96,4.52,3.69,5.88]).reshape(-1,1)
y=numpy.array([0,0,0,0,0,0,1,1,1,1,1,1])

logr=linear_model.LogisticRegression()
logr.fit(x,y)

predicted=logr.predict(numpy.array([3.46]).reshape(-1,1))
print(predicted)

#coefficient
import numpy
from sklearn import linear_model

X=numpy.array([3.78,2.44,2.09,0.14,1.72,1.65,4.92,4.37,4.96,4.52,3.69,5.88]).reshape(-1,1)
y=numpy.array([0,0,0,0,0,0,1,1,1,1,1,1])

logr=linear_model.LogisticRegression()
logr.fit(X,y)

log_odds=logr.coef_
odds=numpy.exp(log_odds)
print(odds)

#probability
import numpy
from sklearn import linear_model

X = numpy.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)
y = numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

logr = linear_model.LogisticRegression()
logr.fit(X,y)

def logit2prob(logr, X):
  log_odds = logr.coef_ * X + logr.intercept_
  odds = numpy.exp(log_odds)
  probability = odds / (1 + odds)
  return(probability)

print(logit2prob(logr, X))