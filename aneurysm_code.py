# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3076160/


# BEST FOR SVM 

# X = rl.drop(['Hospital', 'Row', 'Name', 'Category_AAA', 'AAA_copy','Size_of_aorta','Years_of_HPT_encode', 'Cig_per_day_encode',
#              'Ex_smoker','Years_of_hypertension', 'No_cigs', 'Renal_impairment', 
#              'Dialysis', 'Stroke', 'Lung_disease', 'Family_history_AAA',
#              'Heart_disease', 'Dyslipidemia'], axis = 1)


# BEST FOR MLP CLASSIFIER 

# X = rl.drop(['Hospital', 'Row', 'Name', 'Category_AAA', 'AAA_copy','Size_of_aorta','Years_of_HPT_encode', 'Cig_per_day_encode',
#              'Ex_smoker','Renal_impairment', 'Dialysis', 'Stroke', 'Lung_disease', 'Family_history_AAA',
#              ], axis = 1)


# BEST FOR ADABOOST 

# X = rl.drop(['Hospital', 'Row', 'Name', 'Category_AAA', 'AAA_copy','Size_of_aorta','Years_of_HPT_encode', 'Cig_per_day_encode',
#              'Ex_smoker','Years_of_HPT', 'Cig_per_day', 'Renal_impairment', 
#              'Dialysis', 'Stroke', 'Lung_disease', 'Family_history_AAA',
#              'Heart_disease', 'Dyslipidemia', 'Diabetes', 'Hypertension', 'Smoker', 'Gender'], axis = 1)

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline


import warnings

warnings.filterwarnings('ignore')


data = 'aneurysm_dataset.csv'

df = pd.read_csv(data, header=None)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df.shape

df.head()

col_names = []

for value in df.items():
  title = value[1][2]
  col_names.append(title)

print(col_names)

df.columns = col_names

# drop unwanted rows

rl = df.drop([0,1,2])

# check parameters of data in each column

for col in col_names[3:]:
    
    print(rl[col].value_counts())


# check missing values in variables

rl.isnull().sum()

!pip install category_encoders

# Encode categorical variables with ordinal encoding

import category_encoders as ce

encoder = ce.OrdinalEncoder(cols=['Gender', 'Hypertension', 'Years_of_HPT', 'Dyslipidemia', 'Heart_disease', 
                                  'Family_history_AAA', 'Smoker', 'Cig_per_day', 'Ex_smoker', 'Lung_disease', 
                                  'Stroke','Renal_impairment','Dialysis', 'Diabetes'])

rl = encoder.fit_transform(rl)


rl.isnull().sum()


# OPTIONAL: Find correlation between numerical variables 

# x_corr = rl['AAA_copy'].values
# x_corr = x_corr.astype(float)


# y_corr = rl['SBP'].values
# y_corr = y_corr.astype(float)

# print(x_corr)
# print(y_corr)

# print("SBP")
# print(np.corrcoef(x_corr, y_corr))

# print("DBP")
# z_corr = rl['DBP'].values
# z_corr = z_corr.astype(float)

# print(np.corrcoef(x_corr, z_corr))

# print("Years_of_hypertension")
# h_corr = rl['Years_of_hypertension'].values
# h_corr = h_corr.astype(float)

# print(np.corrcoef(x_corr, h_corr))

# print("Years_of_smoking")
# s_corr = rl['Years_of_smoking'].values
# s_corr = s_corr.astype(float)

# print(np.corrcoef(x_corr, s_corr))

# print("No_cigs")
# c_corr = rl['No_cigs'].values
# c_corr = c_corr.astype(float)

# print(np.corrcoef(x_corr, c_corr))


# # Fill missing data using:

# # OPTION 1: Simple Imputer OR 

# from sklearn.impute import SimpleImputer
# imp = SimpleImputer(missing_values=np.nan, strategy='mean')

# SBP_data = np.array(rl.SBP).reshape(-1,1)

# SBP_data = imp.fit_transform(SBP_data)
# SBP_data = SBP_data.flatten()
# print(SBP_data)
# rl.SBP = SBP_data


# DBP_data = np.array(rl.DBP).reshape(-1,1)

# DBP_data = imp.fit_transform(DBP_data)
# DBP_data = DBP_data.flatten()
# print(DBP_data)
# rl.DBP = DBP_data


# OPTION 2: Iterative Imputer OR 

# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
# imp = IterativeImputer(max_iter=10, random_state=0)

# OPTION 3: KNN Imputer 
from sklearn.impute import KNNImputer
imp = KNNImputer()


# MISSING VALUES FOR SBP

# # Based on AAA_copy (Numerical size of AAA)
# combined_SBP_data = rl.iloc[:, [22,7]]

#Based on Age
combined_SBP_data = rl.iloc[:, [3,7]]

combined_SBP_data = np.array(combined_SBP_data)
combined_SBP_data = imp.fit_transform(combined_SBP_data)

SBP_data = []

for value in combined_SBP_data:
  data = value[1]
  SBP_data.append(data)

print(SBP_data)
rl.SBP = SBP_data


# MISSING VALUES FOR DBP

# Based on AAA_copy (Numerical size of AAA)
combined_DBP_data = rl.iloc[:, [22,8]]

combined_DBP_data = np.array(combined_DBP_data)
combined_DBP_data = imp.fit_transform(combined_DBP_data)

DBP_data = []

for value in combined_DBP_data:
  data = value[1]
  DBP_data.append(data)

print(DBP_data)
rl.DBP = DBP_data


# MISSING VALUES FOR Years_of_smoking

# # Based on AAA_copy (Numerical size of AAA)
# combined_yos_data = rl.iloc[:, [22,14]]

#Based on Smoker
combined_yos_data = rl.iloc[:, [12,14]]

combined_yos_data = np.array(combined_yos_data)
combined_yos_data = imp.fit_transform(combined_yos_data)

yos_data = []

for value in combined_yos_data:
  data = value[1]
  yos_data.append(data)

print(yos_data)
rl.Years_of_smoking = yos_data


# MISSING VALUES FOR Years_of_hypertension

# # Based on AAA_copy (Numerical size of AAA)
# combined_yoh_data = rl.iloc[:, [22,23]]

#Based on Heart_disease
combined_yoh_data = rl.iloc[:, [10,23]]


combined_yoh_data = np.array(combined_yoh_data)
combined_yoh_data = imp.fit_transform(combined_yoh_data)

yoh_data = []

for value in combined_yoh_data:
  data = value[1]
  yoh_data.append(data)

print(yoh_data)
rl.Years_of_hypertension = yoh_data


# MISSING VALUES FOR No_cigs

# Based on AAA_copy (Numerical size of AAA)
# combined_nocig_data = rl.iloc[:, [22,24]]

#Based on Smoker
combined_nocig_data = rl.iloc[:, [12,24]]

combined_nocig_data = np.array(combined_nocig_data)
combined_nocig_data = imp.fit_transform(combined_nocig_data)

nocig_data = []

for value in combined_nocig_data:
  data = value[1]
  nocig_data.append(data)

print(nocig_data)
rl.No_cigs = nocig_data


# Fill missing data using Iterative Imputer

# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
# imp = IterativeImputer(max_iter=10, random_state=0, missing_values = 1.0)

# Fill missing data using KNN Imputer 
from sklearn.impute import KNNImputer
imp = KNNImputer(missing_values=0.0)

#Years_of_HPT

# Based on AAA_copy (Numerical size of AAA)
# combined_yearsofhpt_data = rl.iloc[:, [22,6]]

#Based on Years_of_hypertension
combined_yearsofhpt_data = rl.iloc[:, [23,27]]

combined_yearsofhpt_data = np.array(combined_yearsofhpt_data)
combined_yearsofhpt_data = imp.fit_transform(combined_yearsofhpt_data)

yearsofhpt_data = []

for value in combined_yearsofhpt_data:
  data = value[1]
  yearsofhpt_data.append(data)

print(yearsofhpt_data)
rl.Years_of_HPT_encode = yearsofhpt_data


# Fill missing data using KNN Imputer 
from sklearn.impute import KNNImputer
imp = KNNImputer(missing_values=0.0)

#Cig_per_day

# Based on AAA_copy (Numerical size of AAA)
# combined_cigperday_data = rl.iloc[:, [22,13]]

#Based on No_cigs
combined_cigperday_data = rl.iloc[:, [24,28]]

combined_cigperday_data = np.array(combined_cigperday_data)
combined_cigperday_data = imp.fit_transform(combined_cigperday_data)

cigperday_data = []

for value in combined_cigperday_data:
  data = value[1]
  cigperday_data.append(data)

print(cigperday_data)
rl.Cig_per_day_encode = cigperday_data



# check missing values in variables - must be zero for all 

rl.isnull().sum()


# Standardize numerical data 
from sklearn.preprocessing import StandardScaler


# Standardize Age 
std_scaler = StandardScaler()
age_data = np.array(rl.Age).reshape(-1,1)
age_data = std_scaler.fit_transform(age_data)
age_data = age_data.flatten()

rl.Age = age_data


# Standardize SBP

SBP_data = np.array(rl.SBP).reshape(-1,1)
SBP_data = std_scaler.fit_transform(SBP_data)
SBP_data = SBP_data.flatten()

rl.SBP = SBP_data



# Standardize DBP

DBP_data = np.array(rl.DBP).reshape(-1,1)
DBP_data = std_scaler.fit_transform(DBP_data)
DBP_data = DBP_data.flatten()

rl.DBP = DBP_data


# Standardize Years_of_smoking

yos_data = np.array(rl.Years_of_smoking).reshape(-1,1)
yos_data = std_scaler.fit_transform(yos_data)
yos_data = yos_data.flatten()

rl.Years_of_smoking = yos_data

# Standardize Years_of_hypertension

yoh_data = np.array(rl.Years_of_hypertension).reshape(-1,1)
yoh_data = std_scaler.fit_transform(yoh_data)
yoh_data = yoh_data.flatten()

rl.Years_of_hypertension = yoh_data

# Standardize No_cigs

nocig_data = np.array(rl.No_cigs).reshape(-1,1)
nocig_data = std_scaler.fit_transform(nocig_data)
nocig_data = nocig_data.flatten()

rl.No_cigs = nocig_data

# Check data are all encoded / standardized 

rl.head()


# print(rl)

!pip install imbalanced-learn

## Split data into training and testing sets
## OPTION ONE: KFOLD 

# from sklearn.model_selection import KFold

# kfold = KFold(6, True, 1)

# for train_index, test_index in kfold.split(X, y):
#   X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#   y_train, y_test = y.iloc[train_index], y.iloc[test_index]

## Split data into training and testing sets
## OPTION TWO: STANDARD 

# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42, stratify = y)


## OPTION THREE: STRATIFIED-K-FOLD 

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix


def print_set(set):
  large = 0
  small = 0
  normal = 0
  others = 0

  for value in set:
    if value == 'Large':
      large += 1
    elif value == "Small":
      small += 1
    elif value == "Normal":
      normal += 1
    else:
      others += 1
  total = large + small + normal + others
  result = [large, normal, small, others, total]

  return result 



# Choose data to be included in the train and test 

X = rl.drop(['Hospital', 'Row', 'Name', 'Category_AAA', 'AAA_copy','Size_of_aorta', 'Years_of_HPT_encode', 'Cig_per_day_encode',
             ], axis = 1)

y = rl['Category_AAA']


# Full dataset (including null values)
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=30)

# # Partial dataset (excluding null values)
# skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
accuracy_arr = []

for train_index, test_index in skf.split(X, y):
  X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  y_train, y_test = y.iloc[train_index], y.iloc[test_index]

  print("-----------")
  print("INITIAL TRAIN SET")
  print("large: " + str(print_set(y_train)[0]))
  print("normal: " + str(print_set(y_train)[1]))
  print("small: " + str(print_set(y_train)[2]))
  print("others: " + str(print_set(y_train)[3]))
  print("total: " + str(print_set(y_train)[4]))

  print("-----------")
  print("INITIAL TEST SET")
  print("large: " + str(print_set(y_test)[0]))
  print("normal: " + str(print_set(y_test)[1]))
  print("small: " + str(print_set(y_test)[2]))
  print("others: " + str(print_set(y_test)[3]))
  print("total: " + str(print_set(y_test)[4]))

  incomplete_data = []

  for key, value in X_test["Complete_dataset"].items():
    if value == "0":
      incomplete_data.append(key)

  print("-----------")
  print("INCOMPLTE DATA IN TEST SET")
  print(incomplete_data)

  print("-----------")
  print("NUMBER OF INCOMPLETE DATA")
  switch_total = len(incomplete_data)
  print(switch_total)

  X_test = X_test.drop(incomplete_data)

  y_test = y_test.drop(incomplete_data)

  for value in incomplete_data: 
    data = X.loc[value]
    X_train = X_train.append(data)

  for value in incomplete_data: 
    data = pd.Series([y.loc[value]], index=[value])
    y_train = y_train.append(data)


  print("-----------")
  print("REVISED TEST SET REMOVING INCOMPLETE DATA")
  print("large: " + str(print_set(y_test)[0]))
  print("normal: " + str(print_set(y_test)[1]))
  print("small: " + str(print_set(y_test)[2]))
  print("others: " + str(print_set(y_test)[3]))
  print("total: " + str(print_set(y_test)[4]))

  print("-----------")
  print("REVISED TRAIN SET ADDING INCOMPLETE DATA")
  print("large: " + str(print_set(y_train)[0]))
  print("normal: " + str(print_set(y_train)[1]))
  print("small: " + str(print_set(y_train)[2]))
  print("others: " + str(print_set(y_train)[3]))
  print("total: " + str(print_set(y_train)[4]))

  min_test_set = min(print_set(y_test)[0], print_set(y_test)[1], print_set(y_test)[2])

  large_test = 0
  small_test = 0
  normal_test = 0
  others_test = 0

  shift_data = []

  for key, value in y_test.items():
    if value == 'Large':
      if large_test < min_test_set:
        large_test += 1
      else: 
        shift_data.append(key)

    elif value == "Small":
      if small_test < min_test_set:
        small_test += 1
      else: 
        shift_data.append(key)

    elif value == "Normal":
      if normal_test < min_test_set:
        normal_test += 1
      else: 
        shift_data.append(key)

    else:
      others_test += 1


  for value in shift_data: 
    data = pd.Series([y.loc[value]], index=[value])
    y_train = y_train.append(data)

  y_test = y_test.drop(shift_data)

  X_test = X_test.drop(shift_data)

  for value in shift_data: 
    data = X.loc[value]
    X_train = X_train.append(data)

  print("-----------")
  print("REBALANCED TEST SET")
  print("large: " + str(print_set(y_test)[0]))
  print("normal: " + str(print_set(y_test)[1]))
  print("small: " + str(print_set(y_test)[2]))
  print("others: " + str(print_set(y_test)[3]))
  print("total: " + str(print_set(y_test)[4]))


  print("-----------")
  print("REBALANCED TRAIN SET")
  print("large: " + str(print_set(y_train)[0]))
  print("normal: " + str(print_set(y_train)[1]))
  print("small: " + str(print_set(y_train)[2]))
  print("others: " + str(print_set(y_train)[3]))
  print("total: " + str(print_set(y_train)[4]))


  X_test = X_test.drop(['Complete_dataset'], axis = 1)
  X_train = X_train.drop(['Complete_dataset'], axis = 1)

  print("-----------")
  print("X_TEST")
  print(X_test)

  print("-----------")
  print("Y_TEST")
  print(y_test)

  from imblearn.over_sampling import RandomOverSampler

  ros = RandomOverSampler(random_state=0)
  X_resampled, y_resampled = ros.fit_resample(X_train, y_train)


  print("-----------")
  print("RESAMPLED TRAIN SET")
  print("large: " + str(print_set(y_resampled)[0]))
  print("normal: " + str(print_set(y_resampled)[1]))
  print("small: " + str(print_set(y_resampled)[2]))
  print("others: " + str(print_set(y_resampled)[3]))
  print("total: " + str(print_set(y_resampled)[4]))



# # OPTION 1: RANDOMFOREST CLASSIFIER 
  # from sklearn.ensemble import RandomForestClassifier

# # Partial dataset
#   clf = RandomForestClassifier(n_estimators=600, min_samples_split=2, min_samples_leaf=6, 
#                                max_features=2, max_depth=30, bootstrap='True', 
#                                random_state=20, max_leaf_nodes=4)
  
# Full dataset
  # clf = RandomForestClassifier(n_estimators=100, min_samples_split=2, min_samples_leaf=2, 
  #                              max_features=2, max_depth=30, bootstrap='True', 
  #                              random_state=20, max_leaf_nodes=4)


# OPTION 2: SVC

  from sklearn.pipeline import make_pipeline
  from sklearn.preprocessing import StandardScaler
  from sklearn.svm import SVC

  clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True, random_state = 2))                        
  

# # OPTION 3: ADABOOST 

#   from sklearn.ensemble import AdaBoostClassifier

#   clf = AdaBoostClassifier(random_state=1, n_estimators=10, learning_rate=1)  

# # OPTION 4: MULTI-LAYER PERCEPTION 

#   from sklearn.neural_network import MLPClassifier

#   clf = MLPClassifier(solver='lbfgs', alpha=1.1,
#                       hidden_layer_sizes=(10, 1), random_state=1)

  clf.fit(X_resampled, y_resampled)
  y_pred = clf.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)

  print("-----------")
  print("y_pred")
  print(y_pred)

  proba = clf.predict_proba(X_test)

  print("-----------")
  print("Predict proba: ")
  print(proba)
  
  print("-----------")
  print('Model accuracy score : {0:0.4f}'. format(accuracy))

  accuracy_arr.append(accuracy)

  cm = plot_confusion_matrix(clf, X_test, y_test)

  print("-----------")
  print('CONFUSION MATRIX\n\n', cm)

  from sklearn.metrics import classification_report

  print(classification_report(y_test, y_pred))


print(accuracy_arr)

average_acc = sum(accuracy_arr) / len(accuracy_arr)
print("AVERAGE ACCURACY: " + str(average_acc))


# OPTIONAL: Getting best params for SVC/RandomForest
# clf.get_params()


# n_estimators = np.arange(100, 2000, step=100)
# max_features = ["auto", "sqrt", "log2"]
# max_depth = list(np.arange(10, 100, step=10)) + [None]
# min_samples_split = np.arange(2, 10, step=2)
# min_samples_leaf = [1, 2, 4]
# bootstrap = [True, False]

# param_grid = {
#     "n_estimators": n_estimators,
#     "max_features": max_features,
#     "max_depth": max_depth,
#     "min_samples_split": min_samples_split,
#     "min_samples_leaf": min_samples_leaf,
#     "bootstrap": bootstrap,
# }

# param_grid


# from sklearn.model_selection import RandomizedSearchCV

# random_cv = RandomizedSearchCV(
#     clf, param_grid, n_iter=100, cv=3, scoring="r2", n_jobs=-1
# )


# _ = random_cv.fit(X, y)

# print("Best params:\n")
# print(random_cv.best_params_)
# print(random_cv.best_score_)


# # RESULT: 

# # Best params:

# # {'n_estimators': 1700, 'min_samples_split': 6, 'min_samples_leaf': 4, 'max_features': 'log2', 'max_depth': 30, 'bootstrap': True}
# # -0.0443197860167918


# from sklearn.model_selection import GridSearchCV

# clf = RandomForestClassifier(n_estimators=600, min_samples_split=2, min_samples_leaf=6, 
#                              max_features=2, max_depth=30, bootstrap='True', 
#                              random_state=20, max_leaf_nodes=4)
# new_params = {
#     'min_samples_leaf': [2, 4, 6],
#     'n_estimators': [100, 400, 600, 1000], 
#     'max_features': [2, 4],
#     # 'min_samples_split': [2, 4],
#     # 'max_depth': [20, 30, 40],
#     # 'random_state': [10, 20, 30],
#     # 'max_leaf_nodes': [None, 2, 4, 6, 10],
#     # 'min_impurity_decrease': [0.0, 1.0, 2.0, 3.0]

#     }

# grid_cv = GridSearchCV(clf, param_grid=new_params, scoring='accuracy', n_jobs=-1 )

# grid_cv.fit(X, y)

# print('Best params:\n')
# print(grid_cv.best_params_, '\n')

# print('Best score:\n')
# print(grid_cv.best_score_, '\n')

# print('Best estimator:\n')
# print(grid_cv.best_estimator_, '\n')

# # Best params:

# # {'max_features': 4, 'min_samples_leaf': 8, 'n_estimators': 100, 'max_depth': 30, 'min_samples_split': 2} 

# # Best score:

# # 0.6153846153846153 

# # Best params:

# # {'max_features': 4, 'min_samples_leaf': 6, 'min_samples_split': 2, 'n_estimators': 100} 

# # Best score:

# # 0.6307692307692307 

# FEATURE SCORES FOR RANDOMFOREST CLASSIFIER 

# view the feature scores

# feature_scores = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# feature_scores


# FEATURE SCORES FOR RANDOMFOREST CLASSIFIER 

# # Creating a seaborn bar plot

# sns.barplot(x=feature_scores, y=feature_scores.index)



# # Add labels to the graph

# plt.xlabel('Feature Importance Score')

# plt.ylabel('Features')


# # Add title to the graph

# plt.title("Visualizing Important Features")



# # Visualize the graph

# plt.show()