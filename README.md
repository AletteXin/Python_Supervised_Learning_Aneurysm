# Supervised Learning Aneurysm

## Overview 

Supervised Learning Aneurysm is a research project to develop a supervised learning artificial intelligence model to predict a patient's risk towards Abdominal Aortic Aneurysm ("AAA") using electronic health records. The code is written in Python. 


## Parameters of the project

More than 250 patients were contacted for participation in the retrospective data collection process. 142 of these patients were willing to be interviewed. 

The features considered in the study include:
![Screenshot 2021-10-22 at 11 13 48 AM](https://user-images.githubusercontent.com/85789376/138387538-18e279ef-aa63-43bb-964b-4e4aa8a9b257.png)

Note: All categorical data were encoded with category_encoders, whilst all numerical data were standardized. 

The model was trained to predict the size of a patients' abdominal aorta:
1. Normal (<2.4cm) - extremely low risk of rupture
2. Small (2.4 - 5.4cm) - low risk of rupture 
3. Large (>5.4cm) - high risk of rupture 

Three classifiers were considered throughout the project: RandomForest, SVC and Adaboost 

## Structure of final model 

![Screenshot 2021-10-22 at 11 15 50 AM](https://user-images.githubusercontent.com/85789376/138387709-462d1490-0117-4fd3-add6-3eb697f54d21.png)

## Results of final model 

![Screenshot 2021-10-22 at 11 16 39 AM](https://user-images.githubusercontent.com/85789376/138387801-a255d216-1c00-4efc-9dd5-91bdeb1896fa.png)


## Evolution of thought process

The following was conducted to increase the accuracy of the AI model:

1. GridCV Search was used to inform the parameters of the RandomForestClassifier that will lead to the highest accuracy. 
2. Numerical data generally produced a higher feature score compared to categorical data. As a result, additional numerical features were included, such as Years_of_hypertension and No_cigs.
3. Some data sets had missing values for certain features. A KNN imputer was used to provide substitute numbers. A correlation analysis was conducted betweeen the missing feature and other features. The feature with the highest correlation was used to inform the dummy number used. 
4. Possible subset of features were analysed in a methodical way to understand which combination led to the highest accuracy.
5. Having a two step process allows "Normal" and "Disease" to be classified using one set of feature selection and "Small" and "Large" to be classified using a different feature selection. 
 



