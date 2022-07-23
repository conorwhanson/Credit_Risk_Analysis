# Credit Risk Analysis with Machine Learning

### Purpose & Overview
The purpose of this project was to utilize a variety of machine learning models to predict credit risk.

Machine learning models used:
- Naive random oversampling
- SMOTE oversampling
- Undersampling (using cluster centroids)
- Combo of over/under sampling (aka SMOTEENN)
- Balanced random forest classifier (ensemble method)
- Easy ensemble classifier method (combo of AdaBoost learners and balanced bootstrap samples)

Finanical data from Lending Club was imported, cleaned, encoded, and split into training and testing sets. Next, each model was run with testing and training sets. Finally, the output of each model was run through a confusion matrix and classification report to weigh the effectiveness of each model at predicting credit risk. Note that because high risk credit instances will be far fewer than low risk ones, it is important that each instance of possible high credit risk be categorized with a high level of sensitivity rather than precision. This will tell us that those loans so categorized are highly likely to be correctly categorized. 

### Results

**Method 1: Naive random oversampling (randomly selects minority class data points until classes are balanced)**

Results: **Poor**

![random_oversample_cm](https://github.com/conorwhanson/Credit_Risk_Analysis/blob/main/resources/oversampling_cm.png)

![random_oversample_classification](https://github.com/conorwhanson/Credit_Risk_Analysis/blob/main/resources/oversampling_imb_class.png)

This model did not categorize the high risk loans with adequate sensitivity (recall). Total accuracy was **63 %** with a recall on high risk loans of **57 %**.

**Method 2: SMOTE (synthetic minority oversampling technique; synthetically interpolates data points from minority class to add into that class)**