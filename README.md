# Supervised Learning Aneurysm

## Overview 

Supervised Learning Aneurysm is a research project to develop a supervised learning artificial intelligence model to predict the risk of a patient towards Abdominal Aortic Aneurysm ("AAA") using easily measured health indicators and lifestyle metrics. The code is written in Python. 


## Parameters of the project

More than 300 patients were contacted for participation in the retrospective data collection process. 199 of these patients were willing to be interviewed. 

The features considered in the study include:
1. Age (Numerical)
2. Gender (Categorical)
3. Hypertension (Categorical - Yes or No)
4. Years of Hypertension (Categorical - no_hypertension, short_term (1-10 years), and long_term (>10 years) and Numerical)
5. SBP (Numerical)
6. DBP (Numerical)
7. Dyslipidemia (Categorical - Yes or No)
8. Heart Disease (Categorical - Yes or No)
9. Family history (Categorical - Yes or No)
10. Smoker (Categorical - Yes or No)
11. No. of cigarretes per day (Categorical - non_smoker, light_smoker (1-10), heavy_smoker (>10) and Numerical)
12. Years of smoking (Numerical)
13. Ex smoker (Categorical - never_smoked, Yes, No)
14. Lung disease (Categorical - Yes or No)
15. Stroke (Categorical - Yes or No)
16. Renal impairment (Categorical - Yes or No)
17. Dialysis (Categorical - Yes or No)
18. Diabetes (Categorical - Yes or No)

Note: All categorical data were encoded with category_encoders, whilst all numerical data were standardized. 

The model was trained to predict the output of the size of the abdominal aorta:
1. Normal (<2.4cm) - extremely low risk of rupture
2. Small (2.4 - 5.4cm) - low risk of rupture 
3. Large (>5.4cm) - high risk of rupture 


## Evolution of thought process

The starting code includes a reference to this article (https://www.kaggle.com/prashant111/random-forest-classifier-tutorial) in the creation of a standard RandomForest model. The following was conducted to increase the accuracy of the AI model:

Code
1. The data was stratified into 4 sets, and the average accuracy was calculated. 
2. The number of "Normal", "Small" and "Large" data cases was made equal in both the test and train sets. For the test sets, additional datapoints were fed into the train set. For the train sets, the data was randomly oversampled to meet an equal distribution.
3. GridCV Search was used to inform the parameters of the RandomForestClassifier that will lead to the highest accuracy. 


Data collection 
1. Numerical data generally produced a higher feature score compared to categorical data. As a result, additional numerical features were included, such as Years_of_hypertension and No_cigs.
2. Some data sets had missing values for certain features. An imperative imputer was used to provide best-guessed dummy numbers. A correlation analysis was conducted betweeen the feature and other features. The feature with the highest correlation was used to inform the dummy number used. 

Feature selection 
1. Possible subset of features were analysed in a methodical way to understand which combination led to the highest accuracy. 


## Parameters and characteristics of final model


