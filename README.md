# Initial Contribution Pt 1
The purpose of this project is to first figure out which features best describe the sale price and then train a model to accurately predict the sales price based upon the most relevant features.

![Graph](https://github.com/ebonnecab/mozilla-ds-project/blob/master/graph1.png "Distribution of Sale Price")


## Technologies Used
- Sci-kit learn
- Numpy
- Seaborn
- Pandas
- Python 3
- Jupyter Notebooks

## Summary
There were about 80 possible choices that I had to narrow down to one feature. I did so by, first examining and cleaning the data. Off the bat, there were some features that had too many missing values or didn't have a strong enough relationship with the target to best describe it. 

### Correlation
I started off measuring correlation between the numerical data and the target. A few features stood out such as overall quality and the above ground living area.

![Graph](https://github.com/ebonnecab/mozilla-ds-project/blob/master/graph2.png "Correlation Matrix")

This method gave us some useful information but didn't fully explain the relationship between all the columns and the sales price. 

### Feature Engineering
I split the data into training and testing sets to use a few modules from sklearn to measure feature importance. Because of the mixed datatypes, I split them into categories, did certain datapreprocessing work for different columns and used different techniques to see which feature was holding the most weight. The results are visible in the heatmaps below.

#### Top Multi-variate Features < 25 categories - Mutual Information Regression Module
![Graph](https://github.com/ebonnecab/mozilla-ds-project/blob/master/graph3.png "MI Regression")

#### Top Multi-variate Features < 25 categories - Random Forest Regressor Model
![Graph](https://github.com/ebonnecab/mozilla-ds-project/blob/master/graph4.png "Random Forest Regressor")

#### Top Multi-variate Features > 25 categories - Random Forest Regressor Model
![Graph](https://github.com/ebonnecab/mozilla-ds-project/blob/master/graph5.png "Random Forest Regressor")

#### Top Multi-variate Features > 25 categories - Gradient Boosting Regressor Model
![Graph](https://github.com/ebonnecab/mozilla-ds-project/blob/master/graph6.png "Gradient Boosting Regressor")

## Conclusion
Each of the features that stood out to me were more or less influential but the main feature that seemed to strongly describe the sales price is Overall Quality. I think there are some exceptions in the dataset, but for the most part I see strong growth in the sales price as the quality of the house improves. It may also take in multiple factors that make it a well rounded way to predict how expensive the house will be.

#### Overall Quality x Sales Price
![Graph](https://github.com/ebonnecab/mozilla-ds-project/blob/master/graph7.png "Overall Quality Boxplot")
