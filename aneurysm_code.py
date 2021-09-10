import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

data = 'aneurysm_dataset.csv'

df = pd.read_csv(data, header=None)

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



# Encode categorical variables with ordinal encoding

import category_encoders as ce

encoder = ce.OrdinalEncoder(cols=['Gender', 'Hypertension', 'Years_of_HPT', 'Dyslipidemia', 'Heart_disease', 
                                  'Family_history_AAA', 'Smoker', 'Cig_per_day', 'Lung_disease', 'Stroke','Renal_impairment',
                                  'Dialysis', 'Diabetes', 'Ex_smoker', 'AAA_copy'])


rl = encoder.fit_transform(rl)


# Fill missing data using Iterative Imputer

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(max_iter=10, random_state=0)

combined_SBP_data = rl.iloc[:, [22,7]]

combined_SBP_data = np.array(combined_SBP_data)
combined_SBP_data = imp.fit_transform(combined_SBP_data)

SBP_data = []

for value in combined_SBP_data:
  data = value[1]
  SBP_data.append(data)

print(SBP_data)
rl.SBP = SBP_data



combined_DBP_data = rl.iloc[:, [22,8]]

combined_DBP_data = np.array(combined_DBP_data)
combined_DBP_data = imp.fit_transform(combined_DBP_data)

DBP_data = []

for value in combined_DBP_data:
  data = value[1]
  DBP_data.append(data)

print(DBP_data)
rl.DBP = DBP_data



combined_yos_data = rl.iloc[:, [22,14]]

combined_yos_data = np.array(combined_yos_data)
combined_yos_data = imp.fit_transform(combined_yos_data)

yos_data = []

for value in combined_yos_data:
  data = value[1]
  yos_data.append(data)

print(yos_data)
rl.Years_of_smoking = yos_data


combined_yoh_data = rl.iloc[:, [22,23]]

combined_yoh_data = np.array(combined_yoh_data)
combined_yoh_data = imp.fit_transform(combined_yoh_data)

yoh_data = []

for value in combined_yoh_data:
  data = value[1]
  yoh_data.append(data)

print(yoh_data)
rl.Years_of_hypertension = yoh_data


combined_nocig_data = rl.iloc[:, [22,24]]

combined_nocig_data = np.array(combined_nocig_data)
combined_nocig_data = imp.fit_transform(combined_nocig_data)

nocig_data = []

for value in combined_nocig_data:
  data = value[1]
  nocig_data.append(data)

print(nocig_data)
rl.No_cigs = nocig_data


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


# Choose data to be included in the train and test 

X = rl.drop(['Hospital', 'Row', 'Name', 'Category_AAA', 'AAA_copy', 
             'Years_of_HPT', 'No_cigs'], axis = 1)
print(X)

y = rl['Category_AAA']


## Split data into training and testing sets


from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import plot_confusion_matrix

accuracy_arr = []


skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=32)

for train_index, test_index in skf.split(X, y):
  X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  y_train, y_test = y.iloc[train_index], y.iloc[test_index]

  large_train = 0
  small_train = 0
  normal_train = 0
  others_train = 0

  for value in y_train:
    if value == 'Large':
      large_train += 1
    elif value == "Small":
      small_train += 1
    elif value == "Normal":
      normal_train += 1
    else:
      others_train += 1

  print("-----------")
  print("INITIAL TRAIN SET")
  print("large: " + str(large_train))
  print("normal: " + str(normal_train))
  print("small: " + str(small_train))
  print("others: " + str(others_train))

  large_test = 0
  small_test = 0
  normal_test = 0
  others_test = 0

  for value in y_test:
    if value == 'Large':
      large_test += 1
    elif value == "Small":
      small_test += 1
    elif value == "Normal":
      normal_test += 1
    else:
      others_test += 1

  print("-----------")
  print("INITIAL TEST SET")
  print("large: " + str(large_test))
  print("normal: " + str(normal_test))
  print("small: " + str(small_test))
  print("others: " + str(others_test))

  min_test_set = min(large_test, small_test, normal_test)

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


  large_test = 0
  small_test = 0
  normal_test = 0
  others_test = 0

  for value in y_test:
    if value == 'Large':
      large_test += 1
    elif value == "Small":
      small_test += 1
    elif value == "Normal":
      normal_test += 1
    else:
      others_test += 1

  print("-----------")
  print("REBALANCED TEST SET")
  print("large: " + str(large_test))
  print("normal: " + str(normal_test))
  print("small: " + str(small_test))
  print("others: " + str(others_test))

  large_train = 0
  small_train = 0
  normal_train = 0
  others_train = 0

  for value in y_train:
    if value == 'Large':
      large_train += 1
    elif value == "Small":
      small_train += 1
    elif value == "Normal":
      normal_train += 1
    else:
      others_train += 1

  print("-----------")
  print("REBALANCED TRAIN SET")
  print("large: " + str(large_train))
  print("normal: " + str(normal_train))
  print("small: " + str(small_train))
  print("others: " + str(others_train))

  from imblearn.over_sampling import RandomOverSampler

  ros = RandomOverSampler(random_state=0)
  X_resampled, y_resampled = ros.fit_resample(X_train, y_train)


  large_train = 0
  small_train = 0
  normal_train = 0
  others_train = 0

  for value in y_resampled:
    if value == 'Large':
      large_train += 1
    elif value == "Small":
      small_train += 1
    elif value == "Normal":
      normal_train += 1
    else:
      others_train += 1

  print("-----------")
  print("RESAMPLED TRAIN SET")
  print("large: " + str(large_train))
  print("normal: " + str(normal_train))
  print("small: " + str(small_train))
  print("others: " + str(others_train))

  clf = RandomForestClassifier(n_estimators=1000, min_samples_split=2, min_samples_leaf=4, 
                               max_features=2, max_depth=30, bootstrap='True')
  clf.fit(X_resampled, y_resampled)
  y_pred = clf.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  
  print("-----------")
  print('Model accuracy score variable removed : {0:0.4f}'. format(accuracy))

  accuracy_arr.append(accuracy)

  cm = plot_confusion_matrix(clf, X_test, y_test)

  print('Confusion matrix\n\n', cm)

  from sklearn.metrics import classification_report

  print(classification_report(y_test, y_pred))


print(accuracy_arr)

average_acc = sum(accuracy_arr) / len(accuracy_arr)
print("AVERAGE ACCURACY: " + str(average_acc))


# check the shape of X_train and X_test

X_train.shape, X_test.shape

# Using Grid
clf.get_params()

n_estimators = np.arange(100, 2000, step=100)
max_features = ["auto", "sqrt", "log2"]
max_depth = list(np.arange(10, 100, step=10)) + [None]
min_samples_split = np.arange(2, 10, step=2)
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

param_grid = {
    "n_estimators": n_estimators,
    "max_features": max_features,
    "max_depth": max_depth,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
    "bootstrap": bootstrap,
}

param_grid

# from sklearn.model_selection import GridSearchCV

# clf = RandomForestClassifier(n_estimators=600, min_samples_split=4, min_samples_leaf=2, 
#                                max_features=4, max_depth=20, bootstrap='True')

# new_params = {
#     'min_samples_leaf': [2, 4, 6],
#     'n_estimators': [100, 200, 600, 800, 1000], 
#     'max_features': [2, 4],
#     'min_samples_split': [2, 4],
#     'max_depth': [20, 30, 40],
#     }

# grid_cv = GridSearchCV(clf, param_grid=new_params, scoring='accuracy', n_jobs=-1 )

# grid_cv.fit(X, y)

# print('Best params:\n')
# print(grid_cv.best_params_, '\n')

# print('Best score:\n')
# print(grid_cv.best_score_, '\n')

# print('Best estimator:\n')
# print(grid_cv.best_estimator_, '\n')

# view the feature scores

feature_scores = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

feature_scores


# Creating a seaborn bar plot
sns.barplot(x=feature_scores, y=feature_scores.index)

# Add labels to the graph
plt.xlabel('Feature Importance Score')

plt.ylabel('Features')

# Add title to the graph
plt.title("Visualizing Important Features")

# Visualize the graph
plt.show()


# Choose data to be included in the train and test 
# Remove data which is of lowest feature score 

X = rl.drop(['Hospital', 'Row', 'Name','Category_AAA', 'Dialysis', 'Family_history_AAA', 
             'Stroke', 'Renal_impairment', 'AAA_copy', 'Lung_disease', 'Diabetes', 'Heart_disease', 
             'Dyslipidemia', 'Hypertension', 'Ex_smoker',
             'Years_of_HPT', 'No_cigs'], axis = 1)

y = rl['Category_AAA']


skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=32)

accuracy_arr = []

for train_index, test_index in skf.split(X, y):
  X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  y_train, y_test = y.iloc[train_index], y.iloc[test_index]


  large_train = 0
  small_train = 0
  normal_train = 0
  others_train = 0

  for value in y_train:
    if value == 'Large':
      large_train += 1
    elif value == "Small":
      small_train += 1
    elif value == "Normal":
      normal_train += 1
    else:
      others_train += 1

  print("-----------")
  print("INITIAL TRAIN SET")
  print("large: " + str(large_train))
  print("small: " + str(small_train))
  print("normal: " + str(normal_train))
  print("others: " + str(others_train))

  large_test = 0
  small_test = 0
  normal_test = 0
  others_test = 0

  for value in y_test:
    if value == 'Large':
      large_test += 1
    elif value == "Small":
      small_test += 1
    elif value == "Normal":
      normal_test += 1
    else:
      others_test += 1

  print("-----------")
  print("INITIAL TEST SET")
  print("large: " + str(large_test))
  print("small: " + str(small_test))
  print("normal: " + str(normal_test))
  print("others: " + str(others_test))

  min_test_set = min(large_test, small_test, normal_test)

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


  large_test = 0
  small_test = 0
  normal_test = 0
  others_test = 0

  for value in y_test:
    if value == 'Large':
      large_test += 1
    elif value == "Small":
      small_test += 1
    elif value == "Normal":
      normal_test += 1
    else:
      others_test += 1

  print("-----------")
  print("REBALANCED TEST SET")
  print("large: " + str(large_test))
  print("small: " + str(small_test))
  print("normal: " + str(normal_test))
  print("others: " + str(others_test))

  large_train = 0
  small_train = 0
  normal_train = 0
  others_train = 0

  for value in y_train:
    if value == 'Large':
      large_train += 1
    elif value == "Small":
      small_train += 1
    elif value == "Normal":
      normal_train += 1
    else:
      others_train += 1

  print("-----------")
  print("REBALANCED TRAIN SET")
  print("large: " + str(large_train))
  print("small: " + str(small_train))
  print("normal: " + str(normal_train))
  print("others: " + str(others_train))

  from imblearn.over_sampling import RandomOverSampler

  ros = RandomOverSampler(random_state=0)
  X_resampled, y_resampled = ros.fit_resample(X_train, y_train)


  large_train = 0
  small_train = 0
  normal_train = 0
  others_train = 0

  for value in y_resampled:
    if value == 'Large':
      large_train += 1
    elif value == "Small":
      small_train += 1
    elif value == "Normal":
      normal_train += 1
    else:
      others_train += 1

  print("-----------")
  print("RESAMPLED TRAIN SET")
  print("large: " + str(large_train))
  print("small: " + str(small_train))
  print("normal: " + str(normal_train))
  print("others: " + str(others_train))

  

  clf = RandomForestClassifier(n_estimators=1000, min_samples_split=2, min_samples_leaf=4, 
                               max_features=2, max_depth=30, bootstrap='True')
  clf.fit(X_resampled, y_resampled)
  y_pred = clf.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  
  print("-----------")
  print('Model accuracy score variable removed : {0:0.4f}'. format(accuracy))

  accuracy_arr.append(accuracy)

  cm = plot_confusion_matrix(clf, X_test, y_test)

  print('Confusion matrix\n\n', cm)

  from sklearn.metrics import classification_report

  print(classification_report(y_test, y_pred))


print(accuracy_arr)

average_acc = sum(accuracy_arr) / len(accuracy_arr)
print("AVERAGE ACCURACY: " + str(average_acc))


# view the feature scores

feature_scores = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

feature_scores


# Creating a seaborn bar plot

sns.barplot(x=feature_scores, y=feature_scores.index)

# Add labels to the graph

plt.xlabel('Feature Importance Score')

plt.ylabel('Features')

# Add title to the graph

plt.title("Visualizing Important Features")

# Visualize the graph

plt.show()
