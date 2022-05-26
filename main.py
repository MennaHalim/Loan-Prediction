from builtins import print
from turtle import mode
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, pyplot
from sklearn import preprocessing, tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def RandomForestClassifier_Model():
    # n_estimators = number of decision trees
    RF_model = RandomForestClassifier(n_estimators=35, max_depth=7)
    RF_model.fit(x_train, y_train)
    print("RandomForest Score: ", RF_model.score(x_test, y_test))
    return RF_model

def Decision_Tree_Classifier_Model():
    DT_model = DecisionTreeClassifier()
    DT_model.fit(x_train, y_train)
    print("Decision_Tree Score: ", DT_model.score(x_test, y_test))
"""
    # visualization
    fig = plt.figure(figsize=(25, 20))
    tree.plot_tree(DT_model,
                   feature_names=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome',
                                  'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
                                  'Property_Area'],
                   filled=True);
    fig.savefig("decistion_tree.png", dpi=300)

    # importance
    importance = DT_model.feature_importances_
    # summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
    """

def KNN_Classifier_Model():
    KNN_model = KNeighborsClassifier(n_neighbors=31)
    KNN_model.fit(x_train, y_train)
    print("KNN Score: ", KNN_model.score(x_test, y_test))


def GaussianNB_Classifier_Model():
    NB_model = GaussianNB()
    NB_model.fit(x_train, y_train)
    print("GaussianNB Score: ", NB_model.score(x_test, y_test))


def LogisticRegression_Model():
    LR_model = LogisticRegression(max_iter=1000)
    LR_model.fit(x_train, y_train)
    print("LogisticRegression Score: ", LR_model.score(x_test, y_test))


def SVM():
    SVM_model = LinearSVC(C=0.0001, dual=False)
    SVM_model.fit(x_train, y_train)
    print("SVM Score: ", SVM_model.score(x_test, y_test))


def BaggingClassifier_Model():
    # max_samples: maximum size 0.5=50% of each sample taken from the full dataset
    # max_features: maximum of features 1=100% taken here all 10K
    # n_estimators: number of decision trees
    bagging_model = BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, max_features=1.0, n_estimators=7)
    bagging_model.fit(x_train, y_train)
    print("Bagging Score: ", bagging_model.score(x_test, y_test))



def VotingClassifier_Model():
    # 1) naive bias = NB_model
    # 2) logistic regression =LR_model
    # 3) random forest =LR_model
    # 4) support vector machine = SVM_model
    evc = VotingClassifier(estimators=[('mnb', GaussianNB()), ('lr', LogisticRegression(max_iter=1000)),
                                       ('rf', RandomForestClassifier(n_estimators=35, max_depth=7))], voting='hard')
    evc.fit(x_train, y_train)
    print("voting score: " + str(evc.score(x_test, y_test)))

# read data
data = pd.read_csv('Train data.csv')

# read new customers
new_data = pd.read_csv('New Customer.csv')

# data before preprocessing
#print(data)

# preprocessing
# # Handle  Missing values


# ##Gender
# --> remove non values
#data.dropna(subset=['Gender'], inplace=True)
# --> replace with most repeated value
val = data['Gender'].mode()[0]
data['Gender'].fillna(val, inplace=True)

# ##Married
# --> remove non values
data.dropna(subset=['Married'], inplace=True)

# --> replace with most repeated value
# val = data['Married'].mode()[0]
# data['Married'].fillna(val, inplace=True)

# ##Dependents
# --> replace with 0
#data['Dependents'].fillna(0, inplace=True)
# --> replace with most repeated value
val = data['Dependents'].mode()[0]
data['Dependents'].fillna(val, inplace=True)

# ##Education
# --> remove non values
data.dropna(subset=['Education'], inplace=True)

# --> replace with most repeated value
# val = data['Education'].mode()[0]
# data['Education'].fillna(val, inplace=True)


# ##Self_Employed
# --> remove non values
#data.dropna(subset=['Self_Employed'], inplace=True)

# --> replace with most repeated value
val = data['Self_Employed'].mode()[0]
data['Self_Employed'].fillna(val, inplace=True)

# --> replace with zero
# data['Self_Employed'].fillna(0, inplace=True)


# ##ApplicantIncome
# --> remove non values
#data.dropna(subset=['ApplicantIncome'], inplace=True)

# --> replace with median
val = data['ApplicantIncome'].median()
data['ApplicantIncome'].fillna(val, inplace=True)


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
#data.dropna(subset=['Property_Area'], inplace=True)

# --> replace with most repeated value
val = data['Property_Area'].mode()[0]
data['Property_Area'].fillna(val, inplace=True)

# data after preprocessing
#print(data)

# print before encoding
# print(data)

# # Handle  Wrong format
# --> Encoding
# find important columns name which contain nun numeric values & convert it's type to string
categorical_col = data.select_dtypes(include=['object']).columns.to_list()
categorical_col = categorical_col[1:7]
data[categorical_col] = data[categorical_col].astype('string')

# encode categorical columns
label_encoders = []
for category in categorical_col:
    label_encoder = preprocessing.LabelEncoder()
    data[category] = label_encoder.fit_transform(data[category])
    label_encoders.append(label_encoder)

# print after encoding
# print(data)

# split data into train data and test data
x = data.iloc[:, 1: -1]
y = data.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


Decision_Tree_Classifier_Model()
KNN_Classifier_Model()
GaussianNB_Classifier_Model()
LogisticRegression_Model()
SVM()
BaggingClassifier_Model()
VotingClassifier_Model()
# NN()
RF_model = RandomForestClassifier_Model()

# replace null with most repeated value
new_data = new_data.apply(lambda col: col.fillna(col.value_counts().index[0]))

# encode columns which contain nun numeric values
for idx in range(len(categorical_col)):
    new_data[categorical_col[idx]] = label_encoders[idx].transform(new_data[categorical_col[idx]])


result = RF_model.predict(new_data.iloc[:, 1:])
print(result)

# find encode values for Property_Area = Semiurban & married = Yes
Semiurban_code = label_encoders[-1].transform(['Semiurban'])[0]
married_code = label_encoders[1].transform(['Yes'])[0]
counter = 0

# calculate the percentage of married people in semiurban area that obtained the loan
for idx in range(len(new_data)):
    if result[idx] == 'Y' and new_data['Property_Area'][idx] == Semiurban_code and  new_data['Married'][idx] == married_code:
        counter += 1

print("the percentage of married people in semiurban area that obtained the loan: ", (counter*100)/367)