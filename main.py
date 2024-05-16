import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm 

# Add spam.csv into the code
# This DATASET is being used for TRAINING & TESTING 
spam = pd.read_csv('spam.csv')


z = spam['v2']
y = spam['v1']

# z_train for training inputs, y_train for training labels, z_test for testing inputs, and y_test for testing labels.
# test_size = 0.2 sets the TESTING SET to 20 PERCENT of z and y
z_train, z_test, y_train, y_test = train_test_split(z,y,test_size = 0.2)


cv = CountVectorizer()
features_train = cv.fit_transform(z_train)

model = svm.SVC()
model.fit(features_train, y_train)

features_test = cv.transform(z_test)

print("Accuracy: {}".format(model.score(features_test,y_test)))


# ------- TESTING --------
new_emails = [
    "Congratulations! You've won a free trip to Hawaii. Click here to claim your prize.",
    "Hi, just checking in. Are you available for a meeting tomorrow?",
    "URGENT: Your account has been compromised. Please update your password immediately."
]

# Transform the new email texts using the same CountVectorizer
new_features = cv.transform(new_emails)

# Make predictions using the trained SVM model
predictions = model.predict(new_features)

print("\n---Testing---")
# Print the predictions
for email, prediction in zip(new_emails, predictions):
    print("Email:", email)
    print("Predicted Label:", prediction)
    print()