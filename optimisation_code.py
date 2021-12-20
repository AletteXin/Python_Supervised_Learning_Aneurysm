# OPTIMISATION OF FEATURE SELECTION FOR FIRST LAYER
# FIRST LAYER - SVC (SMOTE)
# SECOND LAYER - SVC (RandomOversampler)

X_drop = rl.drop(['Hospital', 'Row', 'Name', 'Category_AAA', 'AAA_copy','Size_of_aorta', 'Disease',
                 ],axis = 1)

arr = [
           'Heart_disease', 
           'Family_history_AAA', 
           'Lung_disease', 
             'Stroke', 
           'Renal_impairment', 
           'Dialysis', 'Diabetes', 
             'Ex_smoker', 'Smoker', 'Years_of_smoking', 'Cig_per_day', 'No_cigs',
             'Hypertension', 'Years_of_HPT', 'Years_of_hypertension', 
             'Underlying_condition', 'Dyslipidemia', 'Age', 'Gender', 'SBP', 'DBP'
             ]



start = 0
end = start + 1

accuracy_test = []
choices = []

def combinationUtil(arr, data, start,
					end, index, r, empty):
						
	# Current combination is ready
	# to be printed, print it
  choices = []
  all_choices = []
  if (index == r):
    for j in range(r):
      # print(data[j], end = " ");
      choices.append(data[j])
    empty.append(choices)
    return choices;

  i = start;
  array = []
  while(i <= end and end - i + 1 >= r - index):
    data[index] = arr[i];
    result = combinationUtil(arr, data, i + 1,
              end, index + 1, r, empty);
    i += 1;
  return empty;

# Driver Code
r = 9;
n = len(arr);
printCombination(arr, n, r);
type(printCombination(arr, n, r))
choices = printCombination(arr, n, r)

counter = 0

while counter < 210:

  X = X_drop.drop(choices[counter], axis = 1)

  y = rl['Disease']

  # Full dataset (including null values)
  skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=10)

  # # Partial dataset (excluding null values)
  # skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
  accuracy_arr = []
  recall_arr = []
  first_acc_arr = []
  first_recall_arr = [] 
  stepone_TPR = []
  stepone_TNR = []
  stepone_PPV = []
  stepone_NPV = []
  stepone_ACC = []
  stepone_ROC = []
  bothsteps_TPR = []
  bothsteps_TNR = []
  bothsteps_PPV = []
  bothsteps_NPV = []
  bothsteps_ACC = []
  bothsteps_ROC = []

  for train_index, test_index in skf.split(X, y):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


    # REBALANCE TRAINING AND TEST SETS 
    features_first_layer = []

    for key, value in X_test.items():
      features_first_layer.append(key)
    print("FIRST LAYER FEATURES")
    print(features_first_layer)

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

    min_test_set = min(print_disease(y_test)[0], print_disease(y_test)[1])

    disease_test = 0
    normal_test = 0
    others_test = 0

    shift_data = []

    for key, value in y_test.items():
      if value == 'Disease':
        if disease_test < min_test_set:
          disease_test += 1
        else: 
          shift_data.append(key)

      elif value == 'Normal':
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



    X_test = X_test.drop(['Complete_dataset'], axis = 1)
    X_train = X_train.drop(['Complete_dataset'], axis = 1)

  

    # # OPTION 1 FOR OVERSAMPLE: RANDOM OVER SAMPLER

    # from imblearn.over_sampling import RandomOverSampler
    # ros = RandomOverSampler(random_state=0)
    # X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

    # OPTION 2 FOR OVERSAMPLE: SMOTE
    from imblearn.over_sampling import SMOTE, ADASYN
    X_resampled, y_resampled = SMOTE(random_state = 30).fit_resample(X_train, y_train)


    # # OPTION 3 FOR OVERSAMPLE: ADASYN 
    # from imblearn.over_sampling import SMOTE, ADASYN
    # X_resampled, y_resampled = ADASYN(random_state = 0).fit_resample(X_train, y_train)

    print("-----------")
    print("RESAMPLED FIRST LAYER TRAIN SET")
    print("disease: " + str(print_disease(y_resampled)[0]))
    print("normal: " + str(print_disease(y_resampled)[1]))
    print("others: " + str(print_disease(y_resampled)[2]))
    print("total: " + str(print_disease(y_resampled)[3]))

    print("-----------")
    print("RESAMPLED FIRST LAYER TRAIN SET SHAPE")
    print(X_resampled.shape)


  # Features included in second_X_test

    features_first_layer = []

    for key, value in X_test.items():
      features_first_layer.append(key)


    print("FIRST LAYER FEATURES")
    print(features_first_layer)


  # # OPTION 1: RANDOMFOREST CLASSIFIER 
  #   from sklearn.ensemble import RandomForestClassifier

  # # # Partial dataset
  # #   clf = RandomForestClassifier(n_estimators=600, min_samples_split=2, min_samples_leaf=6, 
  # #                                max_features=2, max_depth=30, bootstrap='True', 
  # #                                random_state=20, max_leaf_nodes=4)
    
  # # Full dataset
  #   clf = RandomForestClassifier(n_estimators=1000, min_samples_split=2, min_samples_leaf=4, 
  #                                max_features=4, max_depth=20, bootstrap='True', 
  #                                random_state=20, max_leaf_nodes=2)


  # OPTION 2: SVC

    from sklearn.svm import SVC

    clf = SVC(probability=True, random_state = 2, degree=3, kernel="linear")               
    

  # # OPTION 3: ADABOOST 

  #   from sklearn.ensemble import AdaBoostClassifier

  #   clf = AdaBoostClassifier(random_state=1, n_estimators=100, learning_rate=1)         


  # # OPTION 4: MULTI-LAYER PERCEPTION 

  #   from sklearn.neural_network import MLPClassifier

  #   clf = MLPClassifier(solver='lbfgs', alpha=5.0,
  #                       hidden_layer_sizes=(14, 14), random_state=1)
    
    
    clf.fit(X_resampled, y_resampled)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, pos_label="Disease")

    # from sklearn.metrics import roc_auc_score

    # roc_score = roc_auc_score(y_resampled, clf.predict_proba(X_resampled)[:, 1])
    # print("ROC AUC SCORE")
    # print(roc_score)
    # stepone_ROC.append(roc_score)

    # print("-----------")
    # print("FIRST LAYER PREDICTION")
    # print(y_pred)

    # print("-----------")
    # print("FIRST LAYER GROUND TRUTH")
    # print(y_test)

    # proba = clf.predict_proba(X_test)

    # print("-----------")
    # print("Predict proba: ")
    # print(proba)
    
    print("-----------")
    print('Model accuracy score for first layer : {0:0.4f}'. format(accuracy))

    first_acc_arr.append(accuracy)
    first_recall_arr.append(recall)


  print(first_acc_arr)
  average_acc = sum(first_acc_arr) / len(first_acc_arr)
  append_result = (average_acc, features_first_layer)
  accuracy_test.append(append_result)
  # if end == (len(choices) - 1):
  #   start = 0
  #   counter += 1
  #   end = start + counter
  # else:
  #   start += 1
  counter += 1;
  

print(len(choices))
print(counter)

import heapq

print("TOTAL")
print(len(accuracy_test))
print("TOP TEN")
print(heapq.nlargest(10, accuracy_test))



# OPTIMISATION OF FEATURE SELECTION FOR SECOND LAYER 

# CODE TO OPTIMISE SECOND LAYER 
# FIRST LAYER - SVC (SMOTE)
# SECOND LAYER - SVC (RandomOversampler)

X_drop = rl.drop(['Hospital', 'Row', 'Name', 'Category_AAA', 'AAA_copy','Size_of_aorta', 'Disease','Complete_dataset',
                 ],axis = 1)

arr = [
           'Heart_disease', 
           'Family_history_AAA', 
           'Lung_disease', 
             'Stroke', 
           'Renal_impairment', 
           'Dialysis', 'Diabetes', 
             'Ex_smoker', 'Smoker', 'Years_of_smoking', 'Cig_per_day', 'No_cigs',
             'Hypertension', 'Years_of_HPT', 'Years_of_hypertension', 
             'Underlying_condition', 'Dyslipidemia', 'Age', 'Gender', 'SBP', 'DBP'
             ]



start = 0
end = start + 1

accuracy_test = []
choices = []

def combinationUtil(arr, data, start,
					end, index, r, empty):
						
	# Current combination is ready
	# to be printed, print it
  choices = []
  all_choices = []
  if (index == r):
    for j in range(r):
      # print(data[j], end = " ");
      choices.append(data[j])
    empty.append(choices)
    return choices;

  i = start;
  array = []
  while(i <= end and end - i + 1 >= r - index):
    data[index] = arr[i];
    result = combinationUtil(arr, data, i + 1,
              end, index + 1, r, empty);
    i += 1;
  return empty;

# Driver Code
r = 16;
n = len(arr);
printCombination(arr, n, r);
type(printCombination(arr, n, r))
choices = printCombination(arr, n, r)

counter_run = 0

while counter_run < 5000:

  X = rl.drop(['Hospital', 'Row', 'Name', 'Category_AAA', 'AAA_copy','Size_of_aorta', 'Disease', 
                        'Ex_smoker',  'Dialysis', 'Years_of_HPT',
                        'Stroke', 'Years_of_hypertension',
                        'Years_of_smoking', 'Renal_impairment', 'Underlying_condition'
                        ], axis = 1)

  y = rl['Disease']

  # Full dataset (including null values)
  skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=10)

  # # Partial dataset (excluding null values)
  # skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
  accuracy_arr = []
  recall_arr = []
  first_acc_arr = []
  first_recall_arr = [] 
  stepone_TPR = []
  stepone_TNR = []
  stepone_PPV = []
  stepone_NPV = []
  stepone_ACC = []
  stepone_ROC = []
  bothsteps_TPR = []
  bothsteps_TNR = []
  bothsteps_PPV = []
  bothsteps_NPV = []
  bothsteps_ACC = []
  bothsteps_ROC = []

  for train_index, test_index in skf.split(X, y):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


    # REBALANCE TRAINING AND TEST SETS 
    features_first_layer = []

    for key, value in X_test.items():
      features_first_layer.append(key)
    print("FIRST LAYER FEATURES")
    print(features_first_layer)

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

    min_test_set = min(print_disease(y_test)[0], print_disease(y_test)[1])

    disease_test = 0
    normal_test = 0
    others_test = 0

    shift_data = []

    for key, value in y_test.items():
      if value == 'Disease':
        if disease_test < min_test_set:
          disease_test += 1
        else: 
          shift_data.append(key)

      elif value == 'Normal':
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


    X_test = X_test.drop(['Complete_dataset'], axis = 1)
    X_train = X_train.drop(['Complete_dataset'], axis = 1)

    # # OPTION 1 FOR OVERSAMPLE: RANDOM OVER SAMPLER

    # from imblearn.over_sampling import RandomOverSampler
    # ros = RandomOverSampler(random_state=0)
    # X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

    # OPTION 2 FOR OVERSAMPLE: SMOTE
    from imblearn.over_sampling import SMOTE, ADASYN
    X_resampled, y_resampled = SMOTE(random_state = 30).fit_resample(X_train, y_train)


    # # OPTION 3 FOR OVERSAMPLE: ADASYN 
    # from imblearn.over_sampling import SMOTE, ADASYN
    # X_resampled, y_resampled = ADASYN(random_state = 0).fit_resample(X_train, y_train)

    print("-----------")
    print("RESAMPLED FIRST LAYER TRAIN SET")
    print("disease: " + str(print_disease(y_resampled)[0]))
    print("normal: " + str(print_disease(y_resampled)[1]))
    print("others: " + str(print_disease(y_resampled)[2]))
    print("total: " + str(print_disease(y_resampled)[3]))

    print("-----------")
    print("RESAMPLED FIRST LAYER TRAIN SET SHAPE")
    print(X_resampled.shape)


  # Features included in second_X_test

    features_first_layer = []

    for key, value in X_test.items():
      features_first_layer.append(key)


    print("FIRST LAYER FEATURES")
    print(features_first_layer)


  # # OPTION 1: RANDOMFOREST CLASSIFIER 
  #   from sklearn.ensemble import RandomForestClassifier

  # # # Partial dataset
  # #   clf = RandomForestClassifier(n_estimators=600, min_samples_split=2, min_samples_leaf=6, 
  # #                                max_features=2, max_depth=30, bootstrap='True', 
  # #                                random_state=20, max_leaf_nodes=4)
    
  # # Full dataset
  #   clf = RandomForestClassifier(n_estimators=1000, min_samples_split=2, min_samples_leaf=4, 
  #                                max_features=4, max_depth=20, bootstrap='True', 
  #                                random_state=20, max_leaf_nodes=2)


  # OPTION 2: SVC

    from sklearn.svm import SVC

    clf = SVC(probability=True, random_state = 2, degree=3, kernel="linear")               
    

  # # OPTION 3: ADABOOST 

  #   from sklearn.ensemble import AdaBoostClassifier

  #   clf = AdaBoostClassifier(random_state=1, n_estimators=100, learning_rate=1)         


  # # OPTION 4: MULTI-LAYER PERCEPTION 

  #   from sklearn.neural_network import MLPClassifier

  #   clf = MLPClassifier(solver='lbfgs', alpha=5.0,
  #                       hidden_layer_sizes=(14, 14), random_state=1)
    
    
    clf.fit(X_resampled, y_resampled)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, pos_label="Disease")
    
    print("-----------")
    print('Model accuracy score for first layer : {0:0.4f}'. format(accuracy))

    first_acc_arr.append(accuracy)
    first_recall_arr.append(recall)




  #   # SECOND STEP: PREDICT BETWEEN SMALL AND LARGE 
    second_round_disease_test = []
    second_round_normal_test = []
    second_round_others_test = []

    disease_index = []
    normal_index = []

    # Extracting keys which require second layer (i.e. predicted disease in first layer)

    for index, value in enumerate(y_pred):
      if value == 'Disease':
        disease_index.append(index)
      
      else: 
        normal_index.append(index)

    counter = 0
    
    for key, value in y_test.items():
      if counter in disease_index:
        second_round_disease_test.append(key)
        counter += 1 
        continue
      
      else:
        second_round_normal_test.append(key)
        counter += 1
        continue
      
    
    second_X = X_drop.drop(choices[counter_run], axis = 1)

    second_y = rl['Category_AAA']

    second_X_test = second_X

    second_y_test = second_y

    # To prepare the testing sets for second layer 

    for key, value in second_y_test.items():
      if key in second_round_disease_test:
        continue
      else:
        second_y_test = second_y_test.drop(key)
        second_X_test = second_X_test.drop(key)
        continue

    # To prepare the training sets for second layer 

    second_keys = []
    drop_from_full_set = []

    for key, value in y_train.items():
      second_keys.append(key)

    second_y_train = second_y

    for key, value in second_y_train.items():
      if key in second_keys:
        if value == 'Normal':
          drop_from_full_set.append(key)
          second_y_train = second_y_train.drop(key)
        else:
          continue
      else:
        drop_from_full_set.append(key)
        second_y_train = second_y_train.drop(key)
        continue


    second_X_train = second_X.drop(drop_from_full_set)


  #   print("-----------")
  #   print("SECOND LAYER TRAIN SET")
  #   print("large: " + str(print_set(second_y_train)[0]))
  #   print("small: " + str(print_set(second_y_train)[1]))
  #   print("normal: " + str(print_set(second_y_train)[2]))
  #   print("others: " + str(print_set(second_y_train)[3]))
  #   print("total: " + str(print_set(second_y_train)[4]))


    # OPTION 1 FOR OVERSAMPLE: RANDOM OVER SAMPLER

    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=0)
    second_X_resampled, second_y_resampled = ros.fit_resample(second_X_train, second_y_train)

  #   # # OPTION 2 FOR OVERSAMPLE: SMOTE
  #   # from imblearn.over_sampling import SMOTE, ADASYN
  #   # second_X_resampled, second_y_resampled = SMOTE(random_state = 0).fit_resample(second_X_train, second_y_train)


  #   # # OPTION 3 FOR OVERSAMPLE: ADASYN 
  #   # from imblearn.over_sampling import SMOTE, ADASYN
  #   # second_X_resampled, second_y_resampled = ADASYN(random_state = 10).fit_resample(second_X_train, second_y_train)


  # Features included in second_X_test

    features_second_layer = []

    for key, value in second_X_test.items():
      features_second_layer.append(key)


    print("SECOND LAYER FEATURES")
    print(features_second_layer)


  # # # OPTION 1: RANDOMFOREST CLASSIFIER 
  # #   from sklearn.ensemble import RandomForestClassifier

  # # # # Partial dataset
  # # #   clf = RandomForestClassifier(n_estimators=600, min_samples_split=2, min_samples_leaf=6, 
  # # #                                max_features=2, max_depth=30, bootstrap='True', 
  # # #                                random_state=20, max_leaf_nodes=4)
    
  # # # Full dataset
  # #   clf = RandomForestClassifier(n_estimators=1000, min_samples_split=2, min_samples_leaf=4, 
  # #                                max_features=4, max_depth=20, bootstrap='True', 
  # #                                random_state=20, max_leaf_nodes=2)


  # OPTION 2: SVC

    from sklearn.svm import SVC

    clf = SVC(probability=True, random_state = 2, degree=3, kernel="linear")      

  # # # OPTION 3: ADABOOST

  # #   from sklearn.ensemble import AdaBoostClassifier

  # #   clf = AdaBoostClassifier(random_state=1, n_estimators=100, learning_rate=1)    

    clf.fit(second_X_resampled, second_y_resampled)
    second_y_pred = clf.predict(second_X_test)
    accuracy = accuracy_score(second_y_test, second_y_pred)

    from sklearn.metrics import roc_auc_score

    roc_score = roc_auc_score(second_y_resampled, clf.predict_proba(second_X_resampled)[:, 1])
    print("ROC AUC SCORE")
    print(roc_score)

    bothsteps_ROC.append(roc_score)

    print("-----------")
    print('Model accuracy score for second layer: {0:0.4f}'. format(accuracy))

    print("SECOND LAYER PREDICTION")
    print(second_y_pred)

    print("SECOND LAYER GROUND TRUTH")
    print(second_y_test)

    print("SECOND LAYER X TEST")
    print(second_X_test)

    counter = 0 

    for index, value in enumerate(y_pred):
      if value == 'Disease':
        y_pred[index] = second_y_pred[counter]
        counter += 1
    
    detailed_y_test_keys = []
    
    for key, value in y_test.items():
      detailed_y_test_keys.append(key)

    detailed_y_test = rl['Category_AAA']

    for key, value in detailed_y_test.items():
      if key in detailed_y_test_keys:
        continue
      else:
        detailed_y_test = detailed_y_test.drop(key)
    
    print("GROUND TRUTH")
    print(detailed_y_test)
    print("COMPLETE PREDICTION FROM FIRST AND SECOND LAYER")
    print(y_pred)

    accuracy = accuracy_score(detailed_y_test, y_pred)
    recall = recall_score(detailed_y_test, y_pred, labels=["Large", "Small", "Normal"], average = None)
    
    print("-----------")
    print('Model accuracy score for complete prediction: {0:0.4f}'. format(accuracy))

    accuracy_arr.append(accuracy)
    recall_arr.append(recall)

  # first and second layer combined  
  print(accuracy_arr)
  average_acc = sum(accuracy_arr) / len(accuracy_arr)
  append_result = (average_acc, features_second_layer)
  accuracy_test.append(append_result)

  # if end == (len(choices) - 1):
  #   start = 0
  #   counter += 1
  #   end = start + counter
  # else:
  #   start += 1
  counter_run += 1;
  
