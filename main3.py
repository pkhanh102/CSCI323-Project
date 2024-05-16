# -------- THIS CODE USE NAIVE BAYES ALGORITHM ---------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load the dataset
mail_data = pd.read_csv('mail_data.csv')

# Preprocessing: replacing null values with empty strings
mail_data = mail_data.where(pd.notnull(mail_data), '')

# Set spam = 0 and ham = 1
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

mail_data['Category'] = mail_data['Category'].astype(int)

# Split the data into inputs (X) and labels (Y)
X = mail_data['Message']
Y = mail_data['Category']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8)

# Convert the text data to feature vectors
vectorizer = CountVectorizer()
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# Initialize and train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train_features, Y_train)

# Evaluate the model on the training data
prediction_train = model.predict(X_train_features)
accuracy_train = accuracy_score(Y_train, prediction_train)
print('Accuracy on training data: ', accuracy_train)

# Evaluate the model on the testing data
prediction_test = model.predict(X_test_features)
accuracy_test = accuracy_score(Y_test, prediction_test)
print('\nAccuracy on testing data: ', accuracy_test)


# Predictive System
input_mail = ["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"]

# convert text to feature vectors
input_data_features = vectorizer.transform(input_mail)

# making prediction
prediction = model.predict(input_data_features)

if(prediction):
    print('\nThis email is a HAM email')
else:
    print('\nThis email is a SPAM email')
