import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# read data
data = pd.read_csv('Train data.csv')

# preprocessing
# # Handle  Missing values

# ##Gender
# --> remove non values
data['Gender'].dropna(inplace=True)

# --> replace with most repeated value
# val = data['Gender'].mode()[0]
# data['Gender'].fillna(val, inplace=True)


# ##Married
# --> remove non values
data['Married'].dropna(inplace=True)

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
data['Education'].dropna(inplace=True)

# --> replace with most repeated value
# val = data['Education'].mode()[0]
# data['Education'].fillna(val, inplace=True)


# ##Self_Employed
# --> remove non values
data['Self_Employed'].dropna(inplace=True)

# --> replace with most repeated value
# val = data['Self_Employed'].mode()[0]
# data['Self_Employed'].fillna(val, inplace=True)

# --> replace with zero
# data['Self_Employed'].fillna(0, inplace=True)


# ##ApplicantIncome
# --> remove non values
data['ApplicantIncome'].dropna(inplace=True)

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
data['LoanAmount'].dropna(inplace=True)

# ##Loan_Amount_Term
# --> remove non values
data['Loan_Amount_Term'].dropna(inplace=True)

# ##Credit_History
# --> remove non values
data['Credit_History'].dropna(inplace=True)

# ##Property_Area
# --> remove non values
data['Property_Area'].dropna(inplace=True)

# --> replace with most repeated value
# val = data['Property_Area'].mode()[0]
# data['Property_Area'].fillna(val, inplace=True)


# calculate lift score
from mlxtend.evaluate import lift_score
lift = lift_score(data.iloc[1:, 1], data.iloc[1:, -1])
#lift



# split data into train data and test data
x = data.iloc[:, 1: -1]
y = data.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
