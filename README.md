# Project Title:
Machine Learning Classification Model on Credit Approval dataset
#
# Table of Contents:

  1. Data Cleaning
  2. Exploratory data analysis
  3. Data Preprocessing 
  4. Data Engineering
  5. Model training
  6. Model Evaluation
  7. Conclusion from EDA and preprocessing
  8. Results
  9. Recommendations for future improvement
#
# Project Description:
This is classification problem where we have to predict whether a credit would be approved or not for a client.

Credit Aproval Dataset:

|Field Name|	Order|	Type (Format)|Description|
| -------| -------|-----------|---------|
|checking_status|	1|	string (default)|Status of existing checking account, in Deutsche Mark.|	
|duration	|2|	number (default)	|Duration in months|
|credit_history	|3|	string (default)	|Credit history (credits taken, paid back duly, delays, critical accounts)|
|purpose	|4|	string (default)	|Purpose of the credit (car, television,…)|
|credit_amount	|5|	number (default)	|Credit amount|
|savings_status	|6|	string (default)	|Status of savings account/bonds, in Deutsche Mark.|
|employment	|7|	string (default)	|Present employment, in number of years.|
|installment_commitment	|8|	number (default)|Installment rate in percentage of disposable income|	
|personal_status	|9|	string (default)|Personal status (married, single,…) and sex|
|other_parties	|10|	string (default)|Other debtors / guarantors|	
|residence_since	|11|	number (default)|Present residence since X years|	
|property_magnitude	|12|	string (default)|Property (e.g. real estate)|	
|age	|13|	number (default)	|Age in years|
|other_payment_plans	|14|	string (default)|Other installment plans (banks, stores)|
|housing	|15|	string (default)	|Housing (rent, own,…)|
|existing_credits	|16|	number (default)|Number of existing credits at this bank|	
|job	|17|	string (default)	|Job|
|num_dependents	|18|	number (default)|Number of people being liable to provide maintenance for|	
|own_telephone	|19|	string (default)|Telephone (yes,no)|	
|foreign_worker	|20|	string (default)|Foreign worker (yes,no)|	
accepted	|21|	string (default)	|Class|
#
---
# CONCLUSION from preliminary analysis of a dataset:
---
### Shape of a Dataset:     
Shape of the dataset is 1003 rows x 21 columns.

### NaN values:  
The dataset contains 50 NaN values in column "Age", that should be imputed.

The dataset contains "None" values in 6 columns which are considered as NaN values because their exact meaning is uncertain, and so they should be handled accordingly.
Some of the features contain a high percentage of 'None' values, therefore, I've decided to remove these columns from the dataset.
The ones with lower percentage of 'None' values, will be imputed. 

Imputing of all data is done with KNNImputer in the code below, after encoding the data.  

### Data types:  
Two columns doesnt have correct data types, and they should be converted: "duration" and "accepted".
Since column "accepted" is a target class of our classification problem with binary output it's not necessary to convert the target variable into a string, but I've decided to do it just for clarification and as a practice that needs to be done in further more complicated models. 

### Duplicates:  
There are no duplicate rows in the dataset.

### Typos:       
There are 8 instance typos (categories typos in categorical data), that need to be corrected.

### Descriptive statistics:
The summary statistics for the numerical columns in the dataset shows quick overview of the distribution and data variability that occurs in all columns. The only thing that could be done according to this analysis is scaling of some features to ensure that they are on a similar scale, because some ML algorithms are sensitive to the scale of the features.
#


---
# CONCLUSION from EDA:

---
### Target class: 
The target class is not uniformly distributed, so therefore the best way is to use Stratified K-Folds cross-validaton in order to compute different test scores on different folds of the data.

### Key findings from Histograms:  
There is a exponental distribution detected in "credit_amount" feature and there is a negative skewed distribution detected on features: "age" and "duration".

### Key findings from Boxplots:
Similar to histograms тhere is a skewed distribution with many outliers detected on features: "credit_amount", "age" and "duration" column. Transformation resulted in reducing the skewness in these features. 

### Transformation on the features with skewed distribution:
Transformation are not part of Exploratory Data Analysis, but are done here in order to better analyse post-transformation results. Transformation was made by applying np.sqrt() function on the skewed features. This function was less aggressive in handling outliers compared to logarithmic transformation, which gave me low performance on ML model.

### Key findings from Histplots:
Since this plot was made after transformation, there is reduced skewness in the features showed on these plots.

There is unbalanced dataset in two of the features: 'foreign_worker' and 'num_dependents', but this features were left in the model because it was determined that they don't affect the final score.

### Key findings from Heatmap:
##### Correlation Target - Features: 
Heatmap shows strong positive correlation above 0.7 between target class: 'accepted' and feature: 'credit_amount'. So maybe it is a good idea to give more weight to this feature in order to achive better model performance.

Target class has low correlation with other features, but this is not enough reason to remove these features from the model.  

##### Correlation Feature - Feature:
Also there is no big correlation between the features themselves in the dataset, which suggests that no feature should be removed from the dataset.

### Key findings from Pairplot:
Pairwise relationships between variables in a dataset shows intresting distribution and correlations, especialy between 'credit_amount' and all other features in the dataset. This is sugesting that maybe this feature is KEY PREDICTOR in the predictive model. 

From the pairplot it can be seen that only lower credit amounts were accepted for credit approval, and almost all of the bigger credit amounts were not accepted. 

---

#
#

# RESULTS:
---

After evaluating multiple models on our dataset, I have determined that the best model is RandomForestClassifier(max_depth=10, random_state=42), which consistently outperforms the others in terms of F1 score (macro), Accuracy score, Precision score and Recall score.

#####

## 5 KFold score for RandomForestClassifier:
---
    F1 score (macro):    92.91 %
    Accuracy score:      95.71 %
    Precision score:     91.93 %
    Recall score:        94.00 %

---

#####

## RECOMMENDATIONS FOR FUTURE IMPROVEMENT:
---

In the future, we can focus on the following areas to further enhance our model's performance:

### * Assign more weight to KEY PREDICTOR
By assigning more weight to possible KEY PREDICTOR 'credit_amount' may lead to improved model performance.

### * Feature Engineering: 
Some additional feature transformation could potentially provide more valuable information for the model.

### * Hyperparameter Tuning: 
Conducting a thorough search for optimal hyperparameters can lead to improved model performance.

### * Ensemble Methods: 
Implementing additional ensemble techniques may help capture complex relationships in the data.

By addressing all of these areas, we aim to further enhance the accuracy and reliability of our predictive model.


#
# License
MIT License
#


