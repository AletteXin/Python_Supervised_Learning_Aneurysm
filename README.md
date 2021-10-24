# Supervised Learning Aneurysm

## Overview 

A supervised learning artificial intelligence model to predict a patient's risk towards Abdominal Aortic Aneurysm ("AAA") using electronic health records. The code is written in Python. 


## Data processing

The dataset used contains the electronic health records of 142 patients (due to patient confidentiality reasons, the dataset is not provided in the code). These patients were selected based on availability of data on the size of their abdominal aorta, either from ultrasound scans or CT scans. The patients were interviewed to understand their smoking habits and other underlying chronic conditions. The list of features to be considered was finalised after consultation with a doctor who has experience in treating patients with the disease. 

The features considered in the study include:
![Screenshot 2021-10-22 at 11 13 48 AM](https://user-images.githubusercontent.com/85789376/138387538-18e279ef-aa63-43bb-964b-4e4aa8a9b257.png)

Note: All categorical data were encoded with category_encoders, whilst all numerical data were standardized. 

The model was trained to predict the size of a patients' abdominal aorta:
1. Normal (<2.4cm) - extremely low risk of rupture
2. Small (2.4 - 5.4cm) - low risk of rupture 
3. Large (>5.4cm) - high risk of rupture 

Three classifiers were considered throughout the project: RandomForest, SVC and Adaboost. 

## Structure of final model 

The final model is a 2-step SVC classifier. The dataset was exact steps of the model are depicted in the following figure: 

![Screenshot 2021-10-22 at 11 15 50 AM](https://user-images.githubusercontent.com/85789376/138387709-462d1490-0117-4fd3-add6-3eb697f54d21.png)

## Results of final model 

![Screenshot 2021-10-22 at 11 16 39 AM](https://user-images.githubusercontent.com/85789376/138387801-a255d216-1c00-4efc-9dd5-91bdeb1896fa.png)


## Other notes 

The following was conducted to increase the accuracy of the AI model:

1. GridCV Search was used to inform the parameters of the RandomForestClassifier that will lead to the highest accuracy. 
2. Numerical data generally produced a higher feature score compared to categorical data. As a result, additional numerical features were included, such as Years_of_hypertension and No_cigs.
3. Some data sets had missing values for certain features. A KNN imputer was used to provide substitute numbers. A correlation analysis was conducted betweeen the missing feature and other features. The feature with the highest correlation was used to inform the dummy number used. 
4. Possible subset of features were analysed in a methodical way to understand which combination led to the highest accuracy.
5. Having a two step process allowed "Normal" and "Disease" to be classified using one set of feature selection, and "Small" and "Large" to be classified using a different feature selection. 
 



