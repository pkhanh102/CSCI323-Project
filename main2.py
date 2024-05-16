# -------- THIS CODE USE LOGISTIC REGRESSION ALGORITHM ---------

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- DATA COLLECTION & PREPROCESSING
raw_mail_data = pd.read_csv('mail_data.csv')

mail_data = raw_mail_data.where(pd.notnull(raw_mail_data), '')

#print(mail_data.head(5))
# set spam = 0 & ham = 1
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1


X = mail_data['Message']
Y = mail_data['Category']

# Split data to 2 parts: For training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8)

# Transform test data to feature vectors that can be used as input to the Logistic Regression
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# --- Logistic Regression ---
model = LogisticRegression()

# training the model
model.fit(X_train_features, Y_train)

# prediction on trained data
prediction_training_data = model.predict(X_train_features)
accuracy_training_data = accuracy_score(Y_train, prediction_training_data)
print('Accuracy on training data: ', accuracy_training_data)

# prediction on tested data
prediction_test_data = model.predict(X_test_features)
accuracy_test_data = accuracy_score(Y_test, prediction_test_data)
print('\nAccuracy on testing data: ', accuracy_test_data)


# Predictive System
input_mail = ["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"]

# convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)

# making prediction
prediction = model.predict(input_data_features)

if(prediction):
    print('\nThis email is a HAM email')
else:
    print('\nThis email is a SPAM email')
