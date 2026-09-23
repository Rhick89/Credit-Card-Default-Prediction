# Credit-Card-Default-Prediction
How a data driven model helped identify high-risk credit card customers early - turning a highly imbalanced dataset into an actionable early warning system for defaults

# The Business Problem
A credit card issuer needed a way to flag customers likely to default on their bill before it happened — using account activity, purchase history, and payment behavior already sitting in their systems.

Getting this right matters on both sides of the ledger: missing a defaulter means real financial loss for the lender, while flagging good customers as risky needlessly restricts credit and hurts the relationship. The goal was a model that could reliably separate the two — with a strong bias toward catching actual defaulters, since that's where the cost of a miss is highest.

customer records analyzed - 99,976
raw variables - 36
Proportion of defaulter customers - ~1%

# Class Imbalance
The core challenge wasn't building a model - it was building one that actually worked on a dataset where 99% of customers never default. Left untreated, a model trained on data like this learns the laziest possible trick: predict "no default" for everyone, and still be right 99% of the time on paper - while being useless in practice, since it would catch zero actual defaulters.

That imbalance, plus significant missing data and outliers across the account and transaction fields, meant the real work was in the data preparation and the choice of evaluation metric - not just the algorithm.

# Data Preparation
Variables with more than 30% missing data were dropped rather than imputed, to avoid introducing bias into a study this sensitive to false signals. Remaining gaps were filled using medians (for skewed numeric fields, given the outliers present) and modes (for categorical fields). Outliers were treated using the IQR method, and categorical fields like merchant category and merchant group were one-hot encoded for modelling.

# Exploratory Analysis
Before modelling, the data was profiled to understand what actually separates defaulters from non-defaulters. Some patterns stood out:

1. Defaulters skew younger (average age ~31) vs. non-defaulters (~36)
2. Defaulters take longer to settle bills - a longer average payment span in the months before default
3. Non-defaulters pay off meaningfully larger invoices (40–50% higher) - a signal of financial headroom
4. Entertainment and Clothing & Shoes were both the most-used and most-defaulted merchant categories; Leisure and Food & Beverage stood out as categories where defaulters were disproportionately represented

A K-Means clustering exercise also surfaced four natural customer segments - from high-spend, reliably-paying customers to a distinct high-risk group marked by low transaction activity and a high rate of missed last bills.

# Modelling
Multiple classification approaches were tested - Logistic Regression, Linear Discriminant Analysis, Gaussian Naive Bayes, K-Nearest Neighbors, and Decision Trees - followed by ensemble methods (Bagging, AdaBoost, Gradient Boosting). Because catching defaulters was the priority, recall (the share of actual defaulters correctly flagged) was treated as the deciding metric - not raw accuracy, which is misleading on imbalanced data like this.

# Why accuracy was the wrong yardstick
A Logistic Regression model reached 98.6% accuracy - yet caught 0% of actual defaulters. This illustrates exactly why, in an imbalanced problem like default prediction, accuracy can be a metric that hides a model with no real business value.

# Treating Class Imbalance
SMOTE (Synthetic Minority Oversampling) was applied to rebalance the training data, giving the models a fair shot at learning what actually distinguishes defaulters. This was the turning point - recall on the best models jumped from single digits into the 60–90% range.

# Results
# Model	                                Accuracy	             Recall	             Precision
Logistic Regression	                    64.8%	                 78%	                   3%
LDA	                                    97.6%	                 9%	                    10%
Gaussian NB + Bagging (final model)	    51.9%	                 88%	                  2.5%
KNN	                                    88.2%	                 30%	                   4%
Decision Tree	                          97.0%	                 13%	                   9%

The final model - Gaussian Naive Bayes, ensembled with Bagging - correctly identified 88% of actual defaulters, a deliberate trade-off: it sacrifices overall accuracy and precision in exchange for catching far more of the customers who genuinely go on to default. In a lending context, that trade-off is usually the right one - the cost of missing a defaulter (unrecovered debt) is far higher than the cost of extra scrutiny on a customer who turns out fine.

In practice, this model would flag roughly 9 out of every 10 customers who go on to default - giving the business a meaningful window to intervene (adjusted credit limits, proactive outreach, revised terms) before the loss occurs.

# Business Takeaways
# Spend patterns matter: higher recent credit card spend correlated with a higher likelihood of eventual default - a useful early flag on its own
# Age and payment speed are leading indicators: younger customers and those who consistently take longer to pay are measurably higher-risk
# Category-level risk varies: Leisure and Food & Beverage spending patterns were disproportionately linked to default, suggesting these could be weighted differently in future scoring
# Segmentation adds a second lens: beyond a single risk score, the four customer clusters give the business a simpler way to communicate risk tiers internally

# Tools and Techniques used
1. Python Pandas
2. NumPy
3. scikit-learn
4. SMOTE (imbalanced-learn)
5. K-Means Clustering
6. Logistic Regression
7. LDA
8. Naive Bayes
9. KNN
10. Decision Trees Bagging
11. AdaBoost
12. Gradient Boosting
13. Matplotlib
14. Seaborn
