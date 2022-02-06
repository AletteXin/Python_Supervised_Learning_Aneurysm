# FINAL CODE (CONTINOUS) THAT CAN BE USED TO TEST PHASE 2 
# FIRST LAYER - SVC (SMOTE)
# SECOND LAYER - SVC (RandomOversampler)

title = "FIRST LAYER: SVC (SMOTE); SECOND LAYER: SVC (RandomOversampler)"

X = rl.drop(['Hospital', 'Row', 'Name', 'Category_AAA', 'AAA_copy','Size_of_aorta', 'Disease', 'Underlying_condition',
             # Change from this row onwards
                      'Ex_smoker', 'Dyslipidemia', 'Heart_disease', 
                      'Lung_disease', 'Stroke', 'Family_history_AAA', 
                      # 'Age', 'Gender', 'Years_of_hypertension', 'SBP', 'DBP', 'Hypertension','Dialysis', 
                      'Years_of_HPT', 'Diabetes',  'No_cigs', 'Cig_per_day','Smoker', 
                      'Years_of_smoking', 'Renal_impairment', 
                        ], axis = 1)

# Fields must be same as X above 
X_phasetwo = new_adf.drop(['Hospital', 'Row', 'Name', 'Category_AAA', 'AAA_copy','Size_of_aorta', 'Disease', 'Underlying_condition',
             # Change from this row onwards
                      'Ex_smoker', 'Dyslipidemia', 'Heart_disease', 
                      'Lung_disease', 'Stroke', 'Family_history_AAA', 
                      # 'Age', 'Gender', 'Years_of_hypertension', 'SBP', 'DBP', 'Hypertension','Dialysis', 
                      'Years_of_HPT', 'Diabetes',  'No_cigs', 'Cig_per_day','Smoker', 
                      'Years_of_smoking', 'Renal_impairment', 
                        ], axis = 1)
              

y = rl['Disease']

y_phasetwo = new_adf['Disease']

# Full dataset (including null values)
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=6)

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

  print("-----------")
  print("INITIAL TRAIN SET")
  print("disease: " + str(print_disease(y_train)[0]))
  print("normal: " + str(print_disease(y_train)[1]))
  print("others: " + str(print_disease(y_train)[2]))
  print("total: " + str(print_disease(y_train)[3]))

  print("-----------")
  print("INITIAL TEST SET")
  print("disease: " + str(print_disease(y_test)[0]))
  print("normal: " + str(print_disease(y_test)[1]))
  print("others: " + str(print_disease(y_test)[2]))
  print("total: " + str(print_disease(y_test)[3]))

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
  print("INITIAL TRAIN SET")
  print("disease: " + str(print_disease(y_train)[0]))
  print("normal: " + str(print_disease(y_train)[1]))
  print("others: " + str(print_disease(y_train)[2]))
  print("total: " + str(print_disease(y_train)[3]))

  print("-----------")
  print("INITIAL TEST SET")
  print("disease: " + str(print_disease(y_test)[0]))
  print("normal: " + str(print_disease(y_test)[1]))
  print("others: " + str(print_disease(y_test)[2]))
  print("total: " + str(print_disease(y_test)[3]))

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

  print("-----------")
  print("REBALANCED TEST SET")
  print("disease: " + str(print_disease(y_test)[0]))
  print("normal: " + str(print_disease(y_test)[1]))
  print("others: " + str(print_disease(y_test)[2]))
  print("total: " + str(print_disease(y_test)[3]))


  print("-----------")
  print("REBALANCED TRAIN SET")
  print("disease: " + str(print_disease(y_train)[0]))
  print("normal: " + str(print_disease(y_train)[1]))
  print("others: " + str(print_disease(y_train)[2]))
  print("total: " + str(print_disease(y_train)[3]))


  X_test = X_test.drop(['Complete_dataset'], axis = 1)
  X_train = X_train.drop(['Complete_dataset'], axis = 1)
  X_phasetwo_copy = X_phasetwo.drop(['Complete_dataset'], axis = 1)

  # # FIRST LAYER: DISEASE OR NORMAL 
  
  # first_y = rl['Disease']

  # # Extracting data for y_test set 

  # data_to_drop_first_set_test = []

  # for key, value in y_train.items():
  #   data_to_drop_first_set_test.append(key)

  # y_test = first_y.drop(data_to_drop_first_set_test)


  # # Extrcting data for y_train set 

  # data_to_drop_first_set_train = []

  # for key, value in y_test.items():
  #   data_to_drop_first_set_train.append(key)

  # y_train = first_y.drop(data_to_drop_first_set_train)



  # print("-----------")
  # print("FIRST LAYER TRAIN SET")
  # print("disease: " + str(print_disease(y_train)[0]))
  # print("normal: " + str(print_disease(y_train)[1]))
  # print("others: " + str(print_disease(y_train)[2]))
  # print("total: " + str(print_disease(y_train)[3]))

  # print("-----------")
  # print("FIRST LAYER TEST SET")
  # print("disease: " + str(print_disease(y_test)[0]))
  # print("normal: " + str(print_disease(y_test)[1]))
  # print("others: " + str(print_disease(y_test)[2]))
  # print("total: " + str(print_disease(y_test)[3]))


  # print("-----------")
  # print("X_TEST")
  # print(X_test)

  # print("-----------")
  # print("Y_TEST")
  # print(y_test)

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
  y_pred = clf.predict(X_phasetwo_copy)
  accuracy = accuracy_score(y_phasetwo, y_pred)
  recall = recall_score(y_phasetwo, y_pred, pos_label="Disease")

  from sklearn.metrics import roc_auc_score

  roc_score = roc_auc_score(y_resampled, clf.predict_proba(X_resampled)[:, 1])
  print("ROC AUC SCORE")
  print(roc_score)
  stepone_ROC.append(roc_score)

  print("-----------")
  print("FIRST LAYER PREDICTION")
  print(y_pred)

  print("-----------")
  print("FIRST LAYER GROUND TRUTH")
  print(y_phasetwo)

  proba = clf.predict_proba(X_phasetwo_copy)

  # print("-----------")
  # print("Predict proba: ")
  # print(proba)
  
  print("-----------")
  print('Model accuracy score for first layer : {0:0.4f}'. format(accuracy))

  first_acc_arr.append(accuracy)
  first_recall_arr.append(recall)

  cm = plot_confusion_matrix(clf, X_phasetwo_copy, y_phasetwo)

  print("-----------")
  print('CONFUSION MATRIX\n\n', cm)

  from sklearn.metrics import classification_report

  print(classification_report(y_phasetwo, y_pred))
  print(classification_report(y_phasetwo, y_pred, output_dict=True))

  from sklearn.metrics import confusion_matrix
  matrix = confusion_matrix(y_phasetwo, y_pred, labels =["Disease", "Normal"])
  tn, fp, fn, tp = confusion_matrix(y_phasetwo, y_pred).ravel()
  print("RAVEL")
  print(tn)
  print(fp)
  print(fn)
  print(tp)

  FP = matrix.sum(axis=0) - np.diag(matrix)  
  FN = matrix.sum(axis=1) - np.diag(matrix)
  TP = np.diag(matrix)
  TN = matrix.sum() - (FP + FN + TP)
  print("MATRIX")
  print(TN)
  print(FP)
  print(FN)
  print(TP)

  # Sensitivity, hit rate, recall, or true positive rate
  TPR = TP/(TP+FN)
  # Specificity or true negative rate
  TNR = TN/(TN+FP) 
  # Precision or positive predictive value
  PPV = TP/(TP+FP)
  # Negative predictive value
  NPV = TN/(TN+FN)
  # Fall out or false positive rate
  FPR = FP/(FP+TN)
  # False negative rate
  FNR = FN/(TP+FN)
  # False discovery rate
  FDR = FP/(TP+FP)

  # Overall accuracy
  ACC = (TP+TN)/(TP+FP+FN+TN)



  print("Sensitivity")
  print(TPR)
  stepone_TPR.append(TPR)
  print("Specificity")
  print(TNR)
  stepone_TNR.append(TNR)
  print("Positive Predictive Value")
  print(PPV)
  stepone_PPV.append(PPV)
  print("Negative Predictive Value")
  print(NPV)
  stepone_NPV.append(NPV)
  print("Accuracy")
  print(ACC)
  stepone_ACC.append(ACC)


  # SECOND STEP: PREDICT BETWEEN SMALL AND LARGE 
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
  
  for key, value in y_phasetwo.items():
    if counter in disease_index:
      second_round_disease_test.append(key)
      counter += 1 
      continue
    
    else:
      second_round_normal_test.append(key)
      counter += 1
      continue
    
  # Redefining the feature selection to predict between small and large 

  second_X = rl.drop(['Hospital', 'Row', 'Name', 'Category_AAA', 'AAA_copy','Size_of_aorta', 'Disease', 'Complete_dataset',
                      # 'Hypertension', 'No_cigs',
                      'Ex_smoker', 'Dyslipidemia', 'Heart_disease', 'Dialysis', 
                      'Lung_disease', 'Stroke', 'Family_history_AAA',  'Age',
                      'Gender', 'Cig_per_day','Smoker', 'Years_of_hypertension', 'SBP', 'DBP',
                      'Years_of_HPT', 'Diabetes',
                      'Years_of_smoking', 'Renal_impairment', 'Underlying_condition'
                      ], axis = 1)

  second_y = rl['Category_AAA']

  second_X_test = new_adf.drop(['Hospital', 'Row', 'Name', 'Category_AAA', 'AAA_copy','Size_of_aorta', 'Disease', 'Complete_dataset',
                      # 'Hypertension', 'No_cigs',
                      'Ex_smoker', 'Dyslipidemia', 'Heart_disease', 'Dialysis', 
                      'Lung_disease', 'Stroke', 'Family_history_AAA',  'Age',
                      'Gender', 'Cig_per_day','Smoker', 'Years_of_hypertension', 'SBP', 'DBP',
                      'Years_of_HPT', 'Diabetes',
                      'Years_of_smoking', 'Renal_impairment', 'Underlying_condition'
                      ], axis = 1)

  second_y_test = new_adf['Category_AAA']

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


  print("-----------")
  print("SECOND LAYER TRAIN SET")
  print("large: " + str(print_set(second_y_train)[0]))
  print("small: " + str(print_set(second_y_train)[1]))
  print("normal: " + str(print_set(second_y_train)[2]))
  print("others: " + str(print_set(second_y_train)[3]))
  print("total: " + str(print_set(second_y_train)[4]))


  # OPTION 1 FOR OVERSAMPLE: RANDOM OVER SAMPLER

  from imblearn.over_sampling import RandomOverSampler
  ros = RandomOverSampler(random_state=0)
  second_X_resampled, second_y_resampled = ros.fit_resample(second_X_train, second_y_train)

  # # OPTION 2 FOR OVERSAMPLE: SMOTE
  # from imblearn.over_sampling import SMOTE, ADASYN
  # second_X_resampled, second_y_resampled = SMOTE(random_state = 0).fit_resample(second_X_train, second_y_train)


  # # OPTION 3 FOR OVERSAMPLE: ADASYN 
  # from imblearn.over_sampling import SMOTE, ADASYN
  # second_X_resampled, second_y_resampled = ADASYN(random_state = 10).fit_resample(second_X_train, second_y_train)

  print("-----------")
  print("RESAMPLED SECOND LAYER TRAIN SET")
  print("large: " + str(print_set(second_y_resampled)[0]))
  print("small: " + str(print_set(second_y_resampled)[1]))
  print("normal: " + str(print_set(second_y_resampled)[2]))
  print("others: " + str(print_set(second_y_resampled)[3]))
  print("total: " + str(print_set(second_y_resampled)[4]))


# Features included in second_X_test

  features_second_layer = []

  for key, value in second_X_test.items():
    features_second_layer.append(key)


  print("SECOND LAYER FEATURES")
  print(features_second_layer)


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

  cm = plot_confusion_matrix(clf, second_X_test, second_y_test)

  print("-----------")
  print('CONFUSION MATRIX\n\n', cm)

  from sklearn.metrics import classification_report

  print(classification_report(second_y_test, second_y_pred))


  # To-do: Merge both datasets and come up with the final accuracy 

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
  
  for key, value in y_phasetwo.items():
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

  # cm = plot_confusion_matrix(clf,X_test, detailed_y_test, labels=["Large", "Small", "Normal"])

  print("-----------")
  print('CONFUSION MATRIX\n\n', cm)

  from sklearn.metrics import classification_report

  print(classification_report(detailed_y_test, y_pred, labels=["Large", "Small", "Normal"]))
  print(classification_report(detailed_y_test, y_pred, labels=["Large", "Small", "Normal"], output_dict=True))

  from sklearn.metrics import confusion_matrix
  matrix = confusion_matrix(detailed_y_test, y_pred, labels=["Large", "Small", "Normal"])

  FP = matrix.sum(axis=0) - np.diag(matrix)  
  FN = matrix.sum(axis=1) - np.diag(matrix)
  TP = np.diag(matrix)
  TN = matrix.sum() - (FP + FN + TP)

  print(FP)
  print(FN)
  print(TP)
  print(TN)

  # Sensitivity, hit rate, recall, or true positive rate
  TPR = TP/(TP+FN)
  # Specificity or true negative rate
  TNR = TN/(TN+FP) 
  # Precision or positive predictive value
  PPV = TP/(TP+FP)
  # Negative predictive value
  NPV = TN/(TN+FN)
  # Fall out or false positive rate
  FPR = FP/(FP+TN)
  # False negative rate
  FNR = FN/(TP+FN)
  # False discovery rate
  FDR = FP/(TP+FP)

  # Overall accuracy
  ACC = (TP+TN)/(TP+FP+FN+TN)

  print("Sensitivity")
  print(TPR)
  bothsteps_TPR.append(TPR)
  print("Specificity")
  print(TNR)
  bothsteps_TNR.append(TNR)
  print("Positive Predictive Value")
  print(PPV)
  bothsteps_PPV.append(PPV)
  print("Negative Predictive Value")
  print(NPV)
  bothsteps_NPV.append(NPV)
  print("Accuracy")
  print(ACC)
  bothsteps_ACC.append(ACC)
