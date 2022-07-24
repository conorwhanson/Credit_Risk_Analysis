# Credit Risk Analysis with Machine Learning

## Purpose & Overview
The purpose of this project was to utilize a variety of machine learning models to predict credit risk.

Machine learning models used:
- Naive random oversampling
- SMOTE oversampling
- Undersampling (using cluster centroids)
- Combo of over/under sampling (aka SMOTEENN)
- Balanced random forest classifier (ensemble method)
- Easy ensemble classifier method (combo of AdaBoost learners and balanced bootstrap samples)

Financial data from Lending Club was imported, cleaned, encoded, and split into training and testing sets. Next, each model was run with testing and training sets. Finally, the output of each model was run through a confusion matrix and classification report to weigh the effectiveness of each model at predicting credit risk. Note that because high-risk credit instances will be far fewer than low-risk loans, it is important that each instance of possible high-risk be classified with a high level of sensitivity rather than precision. This will tell us that those loans so classified are likely to be done so correctly. 

## Results

### Method 1: Naive random oversampling (randomly selects minority class data points until classes are balanced)

Results: **Poor**

Confusion Matrix:

![random_oversample_cm](https://github.com/conorwhanson/Credit_Risk_Analysis/blob/main/resources/oversampling_cm.png)

Classification Report:

![random_oversample_class](https://github.com/conorwhanson/Credit_Risk_Analysis/blob/main/resources/oversampling_imb_class.png)

This model did not classify the high-risk loans with adequate sensitivity (recall). Total accuracy was **63 %** with a recall of **57 %** for high-risk loans.

### Method 2: SMOTE (synthetic minority oversampling technique; synthetically interpolates data points from minority class to add into that class)

Results: **Poor**

Confusion Matrix:

![smote_cm](https://github.com/conorwhanson/Credit_Risk_Analysis/blob/main/resources/smote_oversample_cm.png)

Classification Report:

![smote_class](https://github.com/conorwhanson/Credit_Risk_Analysis/blob/main/resources/smote_oversample_class.png)

This model did not classify the high-risk loans with adequate sensitivity (recall), though it did slightly better than the naive random oversampling method. Total SMOTE accuracy was **63 %** with a recall of **61 %** for high-risk loans.

### Method 3: Cluster centroid undersampling (synthetic points are generated to represent identified clusters of majority class, then the majority class is undersampled to balance with the minority class)

Results: **Poor**

Confusion Matrix:

![cc_cm](https://github.com/conorwhanson/Credit_Risk_Analysis/blob/main/resources/cc_undersampling_cm.png)

Classification Report:

![cc_class](https://github.com/conorwhanson/Credit_Risk_Analysis/blob/main/resources/cc_undersampling_class.png)

This model did not classify the high-risk loans with adequate sensitivity (recall). Total accuracy dropped roughly 10% to **51 %** with a recall of **60 %** for high-risk loans.

### Method 4: SMOTEENN (a combination of oversampling and undersampling; SMOTE is used, then data points are dropped based on their proximity to their nearest neighbors with different classes.)

Results: **Poor**

Confusion Matrix:

![smoteenn_cm](https://github.com/conorwhanson/Credit_Risk_Analysis/blob/main/resources/combo_overunder_cm.png)

Classification Report:

![smoteenn_class](https://github.com/conorwhanson/Credit_Risk_Analysis/blob/main/resources/combo_overunder_class.png)

This model did not classify the high-risk loans with adequate sensitivity (recall). Total accuracy was **62 %** with a recall of **70 %** for high-risk loans.

### Method 5: Balanced random forest (a combination of oversampling and undersampling; SMOTE is used, then data points are dropped based on their proximity to their nearest neighbors with different classes.)

Results: **Better**

Confusion Matrix:

![brf_cm](https://github.com/conorwhanson/Credit_Risk_Analysis/blob/main/resources/brf_cm.png)

Classification Report:

![brf_class](https://github.com/conorwhanson/Credit_Risk_Analysis/blob/main/resources/brf_classification.png)

Important features:

![important_features](https://github.com/conorwhanson/Credit_Risk_Analysis/blob/main/resources/important_features.png)

This model did not classify the high-risk loans with adequate sensitivity (recall). Total accuracy was **78 %** (better than all previous models) with a recall of **67 %** for high-risk loans. The recall for low-risk loans was quite high at **91 %**. This recall in conjunction with the recall for high-risk is looking much better.

### Method 6: Easy ensemble classifier with AdaBoost

Results: **Good**

Confusion Matrix:

![eec_adaboost_cm](https://github.com/conorwhanson/Credit_Risk_Analysis/blob/main/resources/easy_ensemble_adaboost_cm.png)

Classification Report:

![eec_adaboost_class](https://github.com/conorwhanson/Credit_Risk_Analysis/blob/main/resources/easy_ensemble_adaboost_class.png)

This model classified the high-risk loans with adequate sensitivity (recall). Total accuracy was **93 %** with a recall of **91 %** for high-risk loans. The recall for low-risk loans was quite high at **94 %** which is the highest thus far among the models. This model did a very good job catching all but 8 high-risk loans.

## Summary & Recommendations

Because the data set was imbalanced the models had a difficult time accurately classifying and predicting high-risk credit risk. Most models, thanks to the abundance of low-risk data to be sampled, had no problem accurately classifying the low-risk credit. However, many of the models had great trouble with both precision and recall for high-risk loans. For example, the first three models (naive oversampling, SMOTE, and cluster centroid undersampling) never cleared over 62 % recall, which given the small set of high-risk loans relative to total loans is cause for alarm. The other resample model, SMOTEENN, faired slightly better with a recall of 70 %.

The ensemble classifiers did remarkably better at classifying high-risk loans. The balanced random forest model did significantly better in recall with 70 % and slightly better precision than the resampling models. However, the easy ensemble classifier with AdaBoost was the model that outperformed all the rest. It had the highest precision (7 %), the highest recall (91 %), and the highest accuracy (93 %) regarding high-risk loans. 

I would recommend implementing the easy ensemble classifier to detect credit risk, although with a caveat. This model may have the tendency to over-classify loan risk, resulting in actual low-risk borrowers being denied a loan. I also recommend seeking to either minimize this as much as possible, or at the very least know the reality of the compromises of over-classifying or under-classifying.