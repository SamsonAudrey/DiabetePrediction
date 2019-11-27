# We import the libraries needed to read the dataset
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler as Scaler
from sklearn.linear_model import LogisticRegression

# We read the data from the CSV file
dataset = pd.read_csv('tutorial_data_eval_PimaIndiansDiabetes.csv', delimiter=';')

# Because the CSV doesn't contain any header, we add column names 
# using the description from the original dataset website
dataset.columns = ["NumTimesPregnant", "PlasmaGlucose", "BloodPressure","TricepsSkin", "Insulin", "Mass","DiabetePediFunc", "Age", "HasDiabetes"]

# Check the shape of the data: we have 768 rows and 9 columns:
# the first 8 columns are features while the last one
# is the supervised label (1 = has diabetes, 0 = no diabetes)

# print('----')
# print(dataset.shape)# (768, 9)
# print(dataset.head())
#    NumTimesPregnant  PlasmaGlucose  BloodPressure  TricepsSkin  Insulin  Mass  DiabetePediFunc  Age  HasDiabetes
# 0                 6            148             72           35        0  33.6            0.627   50            1
# 1                 1             85             66           29        0  26.6            0.351   31            0
# 2                 8            183             64            0        0  23.3            0.672   32            1
# 3                 1             89             66           23       94  28.1            0.167   21            0
# 4                 0            137             40           35      168  43.1            2.288   33            1

# dataset.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 768 entries, 0 to 767
# Data columns (total 9 columns):
# NumTimesPregnant    768 non-null int64
# PlasmaGlucose       768 non-null int64
# BloodPressure       768 non-null int64
# TricepsSkin         768 non-null int64
# Insulin             768 non-null int64
# Mass                768 non-null float64
# DiabetePediFunc     768 non-null float64
# Age                 768 non-null int64
# HasDiabetes         768 non-null int64
# dtypes: float64(2), int64(7)
# memory usage: 54.1 KB

# Correlation Matrix
corr = dataset.corr()
# print('----')
# print(corr)

#                   NumTimesPregnant  PlasmaGlucose  BloodPressure  TricepsSkin   Insulin      Mass  DiabetePediFunc       Age  HasDiabetes
# NumTimesPregnant          1.000000       0.129459       0.141282    -0.081672 -0.073535  0.017683        -0.033523  0.544341     0.221898
# PlasmaGlucose             0.129459       1.000000       0.152590     0.057328  0.331357  0.221071         0.137337  0.263514     0.466581
# BloodPressure             0.141282       0.152590       1.000000     0.207371  0.088933  0.281805         0.041265  0.239528     0.065068
# TricepsSkin              -0.081672       0.057328       0.207371     1.000000  0.436783  0.392573         0.183928 -0.113970     0.074752
# Insulin                  -0.073535       0.331357       0.088933     0.436783  1.000000  0.197859         0.185071 -0.042163     0.130548
# Mass                      0.017683       0.221071       0.281805     0.392573  0.197859  1.000000         0.140647  0.036242     0.292695
# DiabetePediFunc          -0.033523       0.137337       0.041265     0.183928  0.185071  0.140647         1.000000  0.033561     0.173844
# Age                       0.544341       0.263514       0.239528    -0.113970 -0.042163  0.036242         0.033561  1.000000     0.238356
# HasDiabetes               0.221898       0.466581       0.065068     0.074752  0.130548  0.292695         0.173844  0.238356     1.000000

# Heatmap of feature (and Diabete) correlations
sns.heatmap(corr,annot = True)
#plt.show()

# Visualising the data
dataset.hist(bins=50, figsize=(20, 15))
#plt.show()

# Data cleaning 
# 
# Calculate the median value for Mass
median_mass = dataset['Mass'].median()
# Substitute it in the Mass column of the
# dataset where values are 0
dataset['Mass'] = dataset['Mass'].replace(
    to_replace=0, value=median_mass)

# Calculate the median value for BloodPressure
median_bloodP = dataset['BloodPressure'].median()
# Substitute it in the BloodPressure column of the
# dataset where values are 0
dataset['BloodPressure'] = dataset['BloodPressure'].replace(
    to_replace=0, value=median_bloodP)

# Calculate the median value for PlasmaGlucose
median_plasmaGlucose = dataset['PlasmaGlucose'].median()
# Substitute it in the PlasmaGlucose column of the
# dataset where values are 0
dataset['PlasmaGlucose'] = dataset['PlasmaGlucose'].replace(
    to_replace=0, value=median_plasmaGlucose)

# Calculate the median value for TricepsSkin
median_tricepsSkin = dataset['TricepsSkin'].median()
# Substitute it in the TricepsSkin column of the
# dataset where values are 0
dataset['TricepsSkin'] = dataset['TricepsSkin'].replace(
    to_replace=0, value=median_tricepsSkin)

# Calculate the median value for Insulin
median_insulin = dataset['Insulin'].median()
# Substitute it in the Insulin column of the
# dataset where values are 0
dataset['Insulin'] = dataset['Insulin'].replace(
    to_replace=0, value=median_insulin)

# Visualising the data
dataset.hist(bins=50, figsize=(20, 15))
#plt.show()

# Splitting dataset
# Split the training dataset in 80% / 20%
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
# Separate labels from the rest of the dataset
train_set_labels = train_set["HasDiabetes"].copy()
train_set = train_set.drop("HasDiabetes", axis=1)

test_set_labels = test_set["HasDiabetes"].copy()
test_set = test_set.drop("HasDiabetes", axis=1)

#print('----')
df = pd.DataFrame(data=train_set)
#print(df.head())

#      NumTimesPregnant  PlasmaGlucose  BloodPressure  TricepsSkin  Insulin  Mass  DiabetePediFunc  Age
# 60                  2             84             72           23     30.5  32.0            0.304   21
# 618                 9            112             82           24     30.5  28.2            1.282   50
# 346                 1            139             46           19     83.0  28.7            0.654   22
# 294                 0            161             50           23     30.5  21.9            0.254   65
# 231                 6            134             80           37    370.0  46.2            0.238   46

# Apply a scaler
scaler = Scaler()
scaler.fit(train_set)
train_set_scaled = scaler.transform(train_set)
test_set_scaled = scaler.transform(test_set)
df = pd.DataFrame(data=train_set_scaled)
#print('----')
# print(df.head())

#          0         1         2         3         4         5         6         7
# 0  0.117647  0.258065  0.489796  0.272727  0.019832  0.282209  0.096499  0.000000
# 1  0.529412  0.438710  0.591837  0.290909  0.019832  0.204499  0.514091  0.483333
# 2  0.058824  0.612903  0.224490  0.200000  0.082933  0.214724  0.245944  0.016667
# 3  0.000000  0.754839  0.265306  0.272727  0.019832  0.075665  0.075149  0.733333
# 4  0.352941  0.580645  0.571429  0.527273  0.427885  0.572597  0.068318  0.416667

# Model
param_grid = {
    'C': [1.0, 10.0, 50.0],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'shrinking': [True, False],
    'gamma': ['auto', 1, 0.1],
    'coef0': [0.0, 0.1, 0.5]
}
model_svc = SVC()
grid_search = GridSearchCV(
    model_svc, param_grid, cv=10, scoring='accuracy')
grid_search.fit(train_set_scaled, train_set_labels)
# Print the bext score found
print(grid_search.best_score_) # 0.7686938127974617

# Apply the parameters to the model and train it
# Create an instance of the algorithm using parameters from best_estimator_ property
svc = grid_search.best_estimator_
# Use the train dataset to train the model
X = train_set_scaled
Y = train_set_labels

# Train the model
#print(svc.fit(X, Y))


# Prediction
# We create a new (fake) person having the three most correated values high
new_df = pd.DataFrame([[6, 168, 72, 35, 0, 43.6, 0.627, 65]])
# We scale those values like the others
new_df_scaled = scaler.transform(new_df)
# We predict the outcome
prediction = svc.predict(new_df_scaled)
# A value of "1" means that this person is likley to have type 2 diabetes
print(prediction) # 1



# print("//////// 2 E TUTO")
# means = np.mean(train_set, axis=0)
# stds = np.std(train_set, axis=0)
# trainData = (train_set - means)/stds
# testData = (test_set - means)/stds

# diabetesCheck = LogisticRegression()
# diabetesCheck.fit(trainData, train_set_labels)
# accuracy = diabetesCheck.score(testData, test_set_labels)
# print("accuracy = ", accuracy * 100, "%") #76.62337662337663 %

# plt.clf()
# coeff = list(diabetesCheck.coef_[0])
# labels = list(trainData.columns)
# features = pd.DataFrame()
# features['Features'] = labels
# features['importance'] = coeff
# features.sort_values(by=['importance'], ascending=True, inplace=True)
# features['positive'] = features['importance'] > 0
# features.set_index('Features', inplace=True)
# features.importance.plot(kind='barh', figsize=(11, 6),color = features.positive.map({True: 'blue', False: 'red'}))
# plt.xlabel('Importance')
# plt.show()
