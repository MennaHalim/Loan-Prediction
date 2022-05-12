import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# read data
data = pd.read_csv('Train data.csv')
# preprocessing
# # Handle  Missing values

# ##Gender
# --> remove non values
data.dropna(subset=['Gender'], inplace=True)

# --> replace with most repeated value
# val = data['Gender'].mode()[0]
# data['Gender'].fillna(val, inplace=True)


# ##Married
# --> remove non values
data.dropna(subset=['Married'], inplace=True)

# --> replace with most repeated value
# val = data['Married'].mode()[0]
# data['Married'].fillna(val, inplace=True)


# ##Dependents
# --> replace with 0
data['Dependents'].fillna(0, inplace=True)

# --> replace with most repeated value
# val = data['Dependents'].mode()[0]
# data['Dependents'].fillna(val, inplace=True)


# ##Education
# --> remove non values
data.dropna(subset=['Education'], inplace=True)

# --> replace with most repeated value
# val = data['Education'].mode()[0]
# data['Education'].fillna(val, inplace=True)


# ##Self_Employed
# --> remove non values
data.dropna(subset=['Self_Employed'], inplace=True)

# --> replace with most repeated value
# val = data['Self_Employed'].mode()[0]
# data['Self_Employed'].fillna(val, inplace=True)

# --> replace with zero
# data['Self_Employed'].fillna(0, inplace=True)


# ##ApplicantIncome
# --> remove non values
data.dropna(subset=['ApplicantIncome'], inplace=True)

# --> replace with median
# val = data['ApplicantIncome'].median()
# data['ApplicantIncome'].fillna(val, inplace=True)


# ##CoapplicantIncome
# --> replace with zero
data['CoapplicantIncome'].fillna(0, inplace=True)

# --> replace with median
# val = data['CoapplicantIncome'].median()
# data['CoapplicantIncome'].fillna(val, inplace=True)


# ##LoanAmount
# --> remove non values
data.dropna(subset=['LoanAmount'], inplace=True)


# ##Loan_Amount_Term
# --> remove non values
data.dropna(subset=['Loan_Amount_Term'], inplace=True)

# ##Credit_History
# --> remove non values
data.dropna(subset=['Credit_History'], inplace=True)

# ##Property_Area
# --> remove non values
data.dropna(subset=['Property_Area'], inplace=True)

# --> replace with most repeated value
# val = data['Property_Area'].mode()[0]
# data['Property_Area'].fillna(val, inplace=True)


# calculate lift score
from mlxtend.evaluate import lift_score
#print(data.to_string())
#lift = lift_score(data.iloc[:, 1], data.iloc[:, -1])
# lift


# # Handle  Wrong format
# --> Encoding
labelEncoder = LabelEncoder()
labelEncoder.fit(data['Gender'])
data['Gender'] = labelEncoder.transform(data['Gender'])

labelEncoder.fit(data['Married'])
data['Married'] = labelEncoder.transform(data['Married'])

labelEncoder.fit(data['Education'])
data["'Education'"] = labelEncoder.transform(data['Education'])

labelEncoder.fit(data['Property_Area'])
data['Property_Area'] = labelEncoder.transform(data['Property_Area'])


# --> Handle Dependents +3 category
Dependents_list = list(data.iloc[:, 3])
print(Dependents_list)
#for i in range(len(Dependents_list)):
#    if Dependents_list[i] != '0' and Dependents_list[i] != '1' and Dependents_list[i] != '2':
#        data._set_value(i, 'Dependents', '3')

#print(list(data.iloc[:, 3]))

labelEncoder.fit(data['Dependents'])
data['Dependents'] = labelEncoder.transform(data['Dependents'])



# split data into train data and test data
x = data.iloc[:, 1: -1]
y = data.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
