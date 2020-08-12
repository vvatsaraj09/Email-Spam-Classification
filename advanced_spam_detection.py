import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

emails = [[0,"Hello,everyone"],[1,"Win Prizes Everyday"],[0,"Call me now"],[0, "Offers ranging from nothing to everything"]]

df  = pd.DataFrame(np.array(emails),columns=['tags','mail_message'])
# print(df)

x_train , x_test , y_train , y_test = train_test_split(df['mail_message'],df['tags'],random_state = 0)
# print(x_train)

print("Number of rows in training set:{}".format(x_train.shape[0]))
count_vectorizer = CountVectorizer()
training_data = count_vectorizer.fit_transform(x_train)
print(count_vectorizer.get_feature_names())

print(training_data)
testing_data = count_vectorizer.transform(x_test)

naive_bayes = MultinomialNB()
naive_bayes.fit(training_data,y_train)
predications = naive_bayes.predict(testing_data)

print("predicts:",predications)
print(format(accuracy_score(y_test,predications)))
