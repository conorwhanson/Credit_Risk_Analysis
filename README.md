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

Confusion Matrix:

![random_oversample_cm](https://github.com/conorwhanson/Credit_Risk_Analysis/blob/main/resources/oversampling_cm.png)

Classification Report:

![random_oversample_classification](https://github.com/conorwhanson/Credit_Risk_Analysis/blob/main/resources/oversampling_imb_class.png)

This model did not categorize the high risk loans with adequate sensitivity (recall). Total accuracy was **63 %** with a recall of **57 %** for high risk loans.

**Method 2: SMOTE (synthetic minority oversampling technique; synthetically interpolates data points from minority class to add into that class)**

Results: **Poor**

Confusion Matrix:

![smote_cm](https://github.com/conorwhanson/Credit_Risk_Analysis/blob/main/resources/smote_oversample_cm.png)

Classification Report:

![smote_class](https://github.com/conorwhanson/Credit_Risk_Analysis/blob/main/resources/smote_oversample_class.png)

This model did not categorize the high risk loans with adequate sensitivity (recall), though it did slightly better than the naive random oversampling method. Total SMOTE accuracy was **63 %** with a recall of **61 %** for high risk loans.

**Method 3: Cluster centroid undersampling (synthetic points are generated to represent identified clusters of majority class, then the majority class is undersampled to balance with the minority class)**

Results: **Poor**

Confusion Matrix:

![cc_cm](https://github.com/conorwhanson/Credit_Risk_Analysis/blob/main/resources/cc_undersampling_cm.png)

Classification Report:

![cc_class](https://github.com/conorwhanson/Credit_Risk_Analysis/blob/main/resources/cc_undersampling_class.png)

This model did not categorize the high risk loans with adequate sensitivity (recall). Total accuracy was **51 %** with a recall of **60 %** for high risk loans.

**Method 4: SMOTEENN (a combination of oversampling and undersampling; SMOTE is used, then data points are dropped based on their proximity to their nearest neighbors with different classes.)**

Results: **Poor**

Confusion Matrix:

![smoteenn_cm](https://github.com/conorwhanson/Credit_Risk_Analysis/blob/main/resources/combo_overunder_cm.png)

Classification Report:

![smoteenn_class](https://github.com/conorwhanson/Credit_Risk_Analysis/blob/main/resources/combo_overunder_class.png)

This model did not categorize the high risk loans with adequate sensitivity (recall). Total accuracy was **62 %** with a recall of **70 %** for high risk loans.