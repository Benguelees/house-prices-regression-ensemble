# House Prices Regression Ensemble

## Author

Bengü Lees

## Project overview

This project is my final solution for the Kaggle House Prices regression competition. The goal was to predict residential house prices using machine-learning regression models.

## Result

* Public leaderboard score: **0.11879**
* Approximate rank at the time of submission: **233**

## Final approach

I trained three models separately:

* Gradient Boosting
* tuned CatBoost
* ElasticNet

I used five-fold cross-validation to create out-of-fold predictions and then blended the final predictions using these weights:

* 22.77% Gradient Boosting
* 43.23% CatBoost
* 34% ElasticNet

The models were trained independently. Only their final predictions were combined.

## Preprocessing

The preprocessing included:

* filling missing numerical values with the median
* filling missing categorical values with the most frequent category
* one-hot encoding categorical features
* standardizing numerical features
* applying `log1p` to strongly skewed numerical features
* applying `log1p` to the target variable, `SalePrice`

## Evaluation

The models were evaluated using Log RMSE.

## Files

* `Bengue_Lees_House_Prices_Final_Solution.ipynb` — complete commented notebook
* `leaderboard_result.png` — screenshot of the leaderboard result

## What I learned

This project helped me practise:

* data preprocessing
* model tuning
* five-fold cross-validation
* out-of-fold predictions
* model comparison
* weighted ensemble blending
