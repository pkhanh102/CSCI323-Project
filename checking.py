import numpy as np
import pandas as pd

raw_mail_data = pd.read_csv('mail_data.csv')
mail_data = raw_mail_data[['Category', 'Message']]
mail_data = raw_mail_data.where(pd.notnull(raw_mail_data), '')

mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

mail_data['Category'] = mail_data['Category'].astype(int)

X = mail_data['Message']
Y = mail_data['Category']

print(X.value_counts) #5571 Rows in the dataset