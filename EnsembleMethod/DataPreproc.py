import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

workclass = np.array([' Private', ' Self-emp-not-inc', ' Self-emp-inc', 
                      ' Federal-gov', ' Local-gov', ' State-gov', ' Without-pay', 
                      ' Never-worked'])
education = np.array([' Bachelors', ' Some-college', ' 11th', ' HS-grad', 
                      ' Prof-school', ' Assoc-acdm', ' Assoc-voc', ' 9th', 
                      ' 7th-8th', ' 12th', ' Masters', ' 1st-4th', ' 10th', 
                      ' Doctorate', ' 5th-6th', ' Preschool'])
marital_status = np.array([' Married-civ-spouse', ' Divorced', ' Never-married', 
                           ' Separated', ' Widowed', ' Married-spouse-absent', 
                           ' Married-AF-spouse'])
occupation = np.array([' Tech-support', ' Craft-repair', ' Other-service', 
                       ' Sales', ' Exec-managerial', ' Prof-specialty', 
                       ' Handlers-cleaners', ' Machine-op-inspct', 
                       ' Adm-clerical', ' Farming-fishing', ' Transport-moving', 
                       ' Priv-house-serv', ' Protective-serv', ' Armed-Forces']);
relationship = np.array([' Wife', ' Own-child', ' Husband', ' Not-in-family', 
                         ' Other-relative', ' Unmarried'])
race = np.array([' White', ' Asian-Pac-Islander', ' Amer-Indian-Eskimo', ' Other', 
                 ' Black'])
sex = np.array([' Female', ' Male'])
native_country = np.array([' United-States', ' Cambodia', ' England', 
                           ' Puerto-Rico', ' Canada', ' Germany', 
                           ' Outlying-US(Guam-USVI-etc)', ' India', ' Japan', 
                           ' Greece', ' South', ' China', ' Cuba', ' Iran', 
                           ' Honduras', ' Philippines', ' Italy', ' Poland', 
                           ' Jamaica', ' Vietnam', ' Mexico', ' Portugal', 
                           ' Ireland', ' France', ' Dominican-Republic', ' Laos', 
                           ' Ecuador', ' Taiwan', ' Haiti', ' Columbia', ' Hungary', 
                           ' Guatemala', ' Nicaragua', ' Scotland', ' Thailand', 
                           ' Yugoslavia', ' El-Salvador', ' Trinadad&Tobago', 
                           ' Peru', ' Hong', ' Holand-Netherlands'])
label = np.array([' <=50K', ' >50K'])

train = pd.read_csv("DataSet/adult.data",header=None)
train.replace(' ?', np.nan, inplace=True)
train = train.dropna(axis = 0)

test = pd.read_csv("DataSet/adult.test",header=None)
test.replace(' ?', np.nan, inplace=True)
test.replace(' <=50K.', ' <=50K', inplace=True)
test.replace(' >50K.', ' >50K', inplace=True)
test = test.dropna(axis = 0)

le_workclass = LabelEncoder()
le_workclass.fit(workclass)
train[1] = le_workclass.transform(train[1])
test[1] = le_workclass.transform(test[1])

le_education = LabelEncoder()
le_education.fit(education)
train[3] = le_education.transform(train[3])
test[3] = le_education.transform(test[3])

le_marital_status = LabelEncoder()
le_marital_status.fit(marital_status)
train[5] = le_marital_status.transform(train[5])
test[5] = le_marital_status.transform(test[5])

le_occupation = LabelEncoder()
le_occupation.fit(occupation)
train[6] = le_occupation.transform(train[6])
test[6] = le_occupation.transform(test[6])

le_relationship = LabelEncoder()
le_relationship.fit(relationship)
train[7] = le_relationship.transform(train[7])
test[7] = le_relationship.transform(test[7])

le_race = LabelEncoder()
le_race.fit(race)
train[8] = le_race.transform(train[8])
test[8] = le_race.transform(test[8])

le_sex = LabelEncoder()
le_sex.fit(sex)
train[9] = le_sex.transform(train[9])
test[9] = le_sex.transform(test[9])

le_native_country = LabelEncoder()
le_native_country.fit(native_country)
train[13] = le_native_country.transform(train[13])
test[13] = le_native_country.transform(test[13])

le_label = LabelEncoder()
le_label.fit(label)
train[14] = le_label.transform(train[14])
test[14] = le_label.transform(test[14])

train = train.values
X_train, y_train = train[:,:14], train[:,14]

test = test.values
X_test, y_test = test[:,:14], test[:,14]

def ret_data():
    return X_train, y_train, X_test, y_test

