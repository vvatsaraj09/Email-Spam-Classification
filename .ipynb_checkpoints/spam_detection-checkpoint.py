import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV

##Step1: Load Dataset
dataframe = pd.read_csv("spam.csv")
print(dataframe.head())
print(dataframe.describe())

##Step2: Split in to Training and Test Data
x = dataframe["EmailText"]
y = dataframe["Label"]

x_train , y_train = x[0:5000],y[0:5000]
x_test , y_test = x[5000:],y[5000:]


##Step3: Extract Features
cv = CountVectorizer()
features = cv.fit_transform(x_train)

##Step4: Build a model
model = svm.SVC()
model.fit(features,y_train)

##Step5: Test Accuracy
features_test = cv.transform(x_test)
print("Accuracy of the model :",model.score(features_test,y_test))



