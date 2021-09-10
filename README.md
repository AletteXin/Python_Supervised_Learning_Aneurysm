# Supervised Learning Aneurysm

## Overview 

Supervised Learning Aneurysm is a research project to develop an artificial intelligence model to predict the risk of a patient towards Abdominal Aortic Aneurysm ("AAA") using easily measured health indicators and lifestyle metrics. 


## Parameters of the project

More than 300 patients were contacted for participation in the retrospective data collection process. 129 of these patients were willing to be interviewed. 

The features considered in the study include:
1. Age (Numerical)
2. Gender (Numerical)
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

The model was trained to predict the output of the size of the abdominal aorta:
1. Normal (<2cm)
2. Small (2.1 - 5.2cm)
3. Large (>5.2cm)


## Evolution of thought process

The starting code includes a reference to this article (https://www.kaggle.com/prashant111/random-forest-classifier-tutorial) in the creation of a standard RandomForest model. 
