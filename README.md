# Credit-Risk-Modelling-Projects

## [Credit Scoring Classification](https://github.com/rahmadi211/Credit-Risk-Modelling-Projects/blob/main/Credit_Score_Classification.ipynb)
(https://github.com/rahmadi211/Credit-Risk-Modelling-Projects/blob/main/Credit_Score_Classification.ipynb)

This is my most comprehensive credit risk data science containing several applications of data cleaning methods, statistical approaches using bivariate analysis for feature selection and the PyCaret to build, tune and finalize an XGBoost classifier
This is a case of multiclass credit scoring taken from [Kaggle](https://www.kaggle.com/datasets/parisrohan/credit-score-classification) which has three classes in the target variable (Poor, Standard and Good). 

The initial raw dataset requires a lot of cleaning, such as in the case of seven columns listed as containing object values even though they were meant to be numeric columns. The parse_number() method from the R programming tidyverse library was applied to address this issue. This is also an unnormalized dataset where columns containing missing values exists that are fully dependent on a non-primary key column. Hence, an aggregate approach using the pandas library group by method is used to fill in missing values using the mean or median of the subset of the column for each unique value of the column it is fully dependent on. Other methods such as forward fill, backward fill (in the case of the Age column for example), linear interpolation, a multivariate approach using the iterative imputer, string splitting and the pandas library explode method for column transformation are also applied. 

After cleaning the data, new columns are created by feature engineering current ones. Some of these new columns take the form of ratio values (such as the newly created Debt_To_Income feature) but the most notable and significant one is the FICO Score which is often used in credit risk modelling to determine the credit reliability of the borrower based on a number of factors such as their credit history age and loan delinquency records. The feature engineered FICO score in this case is based on the partial model and based on the post-analysis of the feature significance plot of the XGBoost Classifier, this feature engineered column is listed as one of the top ten.

In the bivariate analysis step the Chi-Squared Test of Independence is applied to categorical columns in the dataset whilst the ANOVA test is applied on the numeric ones. Features that are not significant at a p-value of 5% are dropped and will not be included in training the classifier (in this case however, all features turns out to be significant even when the level of significance is set at a p-value of 1%. Thus, no features were actually filtered out)

An XGBoost classifier is then created with a test split value of 20%, tuned and finalized before being applied to predict the target variable values of the testing set. The results of the classification metrics of individual classes of the credit score target variable shows that the model is reliable and fit for purpose with high precision scores for higher grade credit score classes and higher recall classes for lower grade credit score classes (implying that the model is less inclined to misclassify lower grade credit scores as higher ones than it is to misclassify high grade credit scores as lower ones. Which is ideal in the case of credit risk modelling as losing money from giving bad loans is more harmful to a credit company's margin than missing out on a profit from failing to issue a good loan)
A more detailed description/explanation of this project is available on Medium can be accessed through this link:

https://medium.com/@ridwan.s/data-cleaning-unnormalized-datasets-using-aggregation-in-the-case-of-multiclass-xgboost-credit-2289ccd42972




## [WOE/IV Credit Risk](https://github.com/rahmadi211/Credit-Risk-Modelling-Projects/blob/main/WOE_IV_Credit_Risk.ipynb)
(https://github.com/rahmadi211/Credit-Risk-Modelling-Projects/blob/main/WOE_IV_Credit_Risk.ipynb)

This is a project involving the use of Weight of Evidence (WOE) and Information Value (IV) as feature selection method through the application of the [OptBinning](http://gnpalencia.org/optbinning/) library in Python and the use of a Logistic Regression based classifier in the case of a sample generic credit risk dataset that can be accessed in [OpenML](https://www.openml.org/search?type=data&sort=runs&id=43454). A detailed explanation of Weight of Evidence (WOE) and Information Value (IV) as well as about the specific project itself can be accessed on Medium through this link:

https://medium.com/@ridwan.s/the-implementation-of-the-weight-of-evidence-and-information-value-feature-selection-in-machine-efadb67a9deb

## [The Use of Ensemble Methods in the Automation of Home Equity Credit Lines Approval](https://github.com/rahmadi211/Credit-Risk-Modelling-Projects/blob/main/WOE_IV_Credit_Risk.ipynb)
(https://github.com/rahmadi211/Credit-Risk-Modelling-Projects/blob/main/WOE_IV_Credit_Risk.ipynb)

The use of Ensemble Methods and the Auto ML libraries that enables their application is widespread including in cases where credit risk modelling is involved. One of those libraries is the AutoSKLearn library, which in this case is applied to build a classifier to automate the process of approving home equity credit lines using the HMEQ dataset that can be readily accessed in [Kaggle](https://www.kaggle.com/datasets/ajay1735/hmeq-data) or alternatively in [OpenML](https://www.openml.org/search?type=data&status=active&id=43337) (such as in this case). Mutual Information is used as a means of feature selection to filter out any insignificant features in the initial dataset and missing values are handled with the application of a simple imputer. The Feature Engineering step involves creating a FICO Score column to improve the performance of the classifer's result. 
