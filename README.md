# ML Regression Problem
The purpose of this project is to first figure out which features best describe the sale price and then train a model to accurately predict the sales price based upon the features.

## Technologies Used
- Sci-kit learn
- Numpy
- Seaborn
- Pandas
- Python 3
- Jupyter Notebooks

## Summary
I am still currently trying to figure out which features are most useful in training the model. I am currently using 51 of the 76 original features. The difficulty lies in the mix of categorical and numerical data and how to use them to predict a continuous value. With the help one hot encoding, I transformed the categorial data into values of 0 and 1 to be used with the model. So far, I've trained a Random Forest Regressor and a Linear Regression model. The RFR model performed OK, but not great as indicated by the R2 score and Mean Squared Error. The LR model did not perform well at all. I suspect issues of **multicollinearity** are affecting both of these models. I'm in the process of getting to the bottom of this and will update the README shortly!
