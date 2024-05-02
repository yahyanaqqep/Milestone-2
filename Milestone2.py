import pickle as pk
import numpy as np
import pandas as pd
import sklearn.preprocessing as spre
import matplotlib.pyplot as plt
import sklearn.tree as sktree
import sklearn.neighbors as knn
import sklearn.svm as svm
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV 
from sklearn.preprocessing import PolynomialFeatures


Dataset_File = pd.read_csv("ElecDeviceRatingPrediction_Milestone2.csv")

Y = Dataset_File['rating']
X = Dataset_File
# --------------------------------------Label Encoding-------------------------------------
brandLE = spre.LabelEncoder()
X['brand'] = brandLE.fit_transform(X['brand'])
# fileHandler = open("brand_Label_Encoder.pk","wb")
# pk.dump(brandLE,fileHandler)
processor_brandLE = spre.LabelEncoder()
X['processor_brand'] = processor_brandLE.fit_transform(X['processor_brand'])
processor_nameLE = spre.LabelEncoder()
X['processor_name'] = processor_nameLE.fit_transform(X['processor_name'])
ram_typeLE = spre.LabelEncoder()
X['ram_type'] = ram_typeLE.fit_transform(X['ram_type'])
osLE = spre.LabelEncoder()
X['os'] = osLE.fit_transform(X['os'])
weightLE = spre.LabelEncoder()
X['weight'] = weightLE.fit_transform(X['weight'])
TouchscreenLE = spre.LabelEncoder()
X['Touchscreen'] = TouchscreenLE.fit_transform(X['Touchscreen'])
msofficeLE = spre.LabelEncoder()
X['msoffice'] = msofficeLE.fit_transform(X['msoffice'])
ratingLE = spre.LabelEncoder()
Y = ratingLE.fit_transform(Y)
X['rating'] = ratingLE.transform(X['rating'])

# --------------------------------------Removing extra text from numerical feilds(GB, th, ...)---------------------------

X['processor_gnrtn'] = [0 if x=='Not Available' else int(x.replace('th','')) for x in X['processor_gnrtn']]

X['ram_gb'] = [int(x.replace('GB','')) for x in X['ram_gb']]

X['ssd'] = [int(x.replace('GB','')) for x in X['ssd']]

X['hdd'] = [int(x.replace('GB','')) for x in X['hdd']]

X['graphic_card_gb'] = [int(x.replace('GB','')) for x in X['graphic_card_gb']]

X['warranty'] = [0 if x=='No warranty' else int(x.replace('year','').replace('s','')) for x in X['warranty']]

# --------------------------------------Getting correlation to do feature selection-------------------------------------
corr = X.corr()
X = X.drop('rating', axis=1)
corr = corr[:]['rating']
corr = corr.abs()
print(corr)
# plt.barh(corr.sort_values().index, corr.sort_values().values)
# plt.show()
# ---------------------------------------Dropping low correlation features----------------------------------------------
cols = ['ram_type', 'os', 'Price', 'hdd', 'weight', 'processor_name','Touchscreen','ram_gb',
        'processor_gnrtn', 'brand', 'graphic_card_gb', 'processor_brand', 'ssd']
X = X.drop(cols,axis=1)
print("number of empty cells: ", X.isna().sum().sum())
polyfeat = PolynomialFeatures(degree=4)
X = polyfeat.fit_transform(X)
# ---------------------------------------Preparing Data---------------------------------------
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2, random_state=10, shuffle=True)
# ---------------------------------------Creating models--------------------------------------
DecTree = sktree.DecisionTreeClassifier(ccp_alpha=0.005, max_depth=115, criterion='gini')
kNearest = knn.KNeighborsClassifier(2, algorithm='brute',p=1 )
supportVector = svm.SVC(C=1, gamma=0.1, kernel='rbf')
naivebayes = BernoulliNB(alpha=10)

DecTree.fit(x_train, y_train)
kNearest.fit(x_train,y_train)
supportVector.fit(x_train,y_train)
naivebayes.fit(x_train, y_train)
# param_grid = {'C': [0.1, 1, 10, 100, 1000],  
#               'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
#               'kernel': ['rbf']}  
  
# grid = GridSearchCV(supportVector, param_grid, refit = True, verbose = 3) 
# ---------------------------------------Checking score---------------------------------------
DecScore = DecTree.score(x_test, y_test)
kScore = kNearest.score(x_test,y_test)
svmScore = supportVector.score(x_test,y_test)
nvbScore = naivebayes.score(x_test,y_test)
# grid.fit(x_train, y_train)
print("Decision Tree Score: ", DecScore*100,"%")
print("K-nearest neighbour Score: ", kScore*100, "%")
print("SVM Score: ", svmScore*100, "%")
print("Bernoulli Score: ", nvbScore*100, "%")
# print(grid.best_params_)
# -----------------------------------trying comitte vote--------------------------------------
CorrectlyClassified = 0
MistakenlyClassified = 0
DecVote = DecTree.predict(x_test)
kVote = kNearest.predict(x_test)
svmVote = supportVector.predict(x_test)
nvbVote = naivebayes.predict(x_test)
for i in range(0,len(x_test)):
    zeroVote = 0
    oneVote = 0
    finalPrediction = None
    if DecVote[i]==0:
        zeroVote+=1 
    else:
        oneVote+=1
    if nvbVote[i]==0:
        zeroVote+=1
    else:
        oneVote+=1
    if svmVote[i]==0:
        zeroVote+=1
    else:
        oneVote+=1
    if zeroVote>oneVote:
        finalPrediction=0
    else:
        finalPrediction=1
    if finalPrediction==y_test[i]:
        CorrectlyClassified+=1
    else:
        MistakenlyClassified+=1
    
print("Committe accuracy: ", (CorrectlyClassified/len(x_test))*100)