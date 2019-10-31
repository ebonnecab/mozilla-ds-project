# Initial Contribution Pt 1
The purpose of this project is to first figure out which features best describe the sale price and then train a model to accurately predict the sales price based upon the most relevant features.

![Graph](https://github.com/ebonnecab/mozilla-ds-project/blob/master/graphs/graph1.png "Distribution of Sale Price")


## Technologies Used
- Sci-kit learn
- Numpy
- Seaborn
- Pandas
- Python 3
- Jupyter Notebooks

## Summary
There were about 80 possible choices that I had to narrow down to one feature. I did so by, first examining and cleaning the data. Off the bat, there were some features that had too many missing values or didn't have a strong enough relationship with the target to best describe it. I eliminated those and proceeded to use other methods to measure the relationship of the features to the sales price.

### Correlation
I started off measuring correlation between the numerical data and the target. A few features stood out such as overall quality and the above ground living area.

![Graph](https://github.com/ebonnecab/mozilla-ds-project/blob/master/graphs/graph2.png "Correlation Matrix")

This method gave us some useful information but didn't fully explain the relationship between all the columns and the sales price. 

### Feature Engineering
I split the data into training and testing sets to use a few modules from sklearn to measure feature importance. Because of the mixed datatypes, I split them into categories and used different techniques to see which feature was holding the most weight. The results are visible in the heatmaps below.

#### Top Multi-variate Features < 25 categories - Mutual Information Regression Module
![Graph](https://github.com/ebonnecab/mozilla-ds-project/blob/master/graphs/graph3.png "MI Regression")

#### Top Multi-variate Features < 25 categories - Random Forest Regressor Model
![Graph](https://github.com/ebonnecab/mozilla-ds-project/blob/master/graphs/graph4.png "Random Forest Regressor")

#### Top Multi-variate Features > 25 categories - Random Forest Regressor Model
![Graph](https://github.com/ebonnecab/mozilla-ds-project/blob/master/graphs/graph5.png "Random Forest Regressor")

#### Top Multi-variate Features > 25 categories - Gradient Boosting Regressor Model
![Graph](https://github.com/ebonnecab/mozilla-ds-project/blob/master/graphs/graph6.png "Gradient Boosting Regressor")

## Initial Conclusion
Each of the features that stood out to me were more or less influential but the main feature that seemed to strongly describe the sales price is Overall Quality. I think there are some exceptions in the dataset, but for the most part I see strong growth in the sales price as the quality of the house improves. I'm not sure what the criteria was for measuring overall quality, but it is probably based upon multiple factors, making it a well rounded way to predict how expensive the house will be.

#### Overall Quality x Sales Price
![Graph](https://github.com/ebonnecab/mozilla-ds-project/blob/master/graphs/graph7.png "Overall Quality Boxplot")

# Second Contribution
The purpose of the second portion of this project was to choose a regression model for this dataset and justify my choice. 

## Summary
Finding the right model to predict the sales price based upon 80 features was a little tricky. Once again, I was faced with the task of deciding which features actually describe the sales price before I could even begin to choose a model. Using sklearn modules, I created a pipeline that returned the most relevant features. I was then able to test them on my models and decide which one was best for this problem.

### Model Selection
I chose six models based upon research, Sklearn documentation, and prior experience. Each came with their own strengths and weaknesses.

**Simple Linear Regression**
- Can be problematic with multivariate predictors
- sensitive to multi-collinearity and outliers

**Lasso**
- Reduces model complexity and prevents overfitting by penalizing the coefficients
- Able to find the most important features and reduce multi-collinearity
- Sometimes it might ignore features you need like grouped variables

**Ridge**
- Also reduces model complexity and prevents overfitting by shrinking the effects of the coeffecients
- Can detect which features are more important but still uses all of them in the final model

**Elastic Net**
- Combines the regularization techniques used in Lasso and Ridge

**Gradient Boosting**
- Builds off the errors of the previous weaker models
- Kind of hard to tune since you have to account for number of trees, depth of trees, and learning rate

**Random Forest**
- Less susceptible to overfitting since each tree is trained independently using a random sample
- Model is biased towards categorical data with more variables

### Preprocessing
I wanted an easier way to preprocess my data so I used Sklearn's Column Transformer to create a custom preprocessor that can handle heterogenous data and process them based upon their datatype.

```
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
categor_cols = X_train.select_dtypes(include=['object']).columns

cat_transformer = Pipeline(
    steps = [
    ('label', MultiColumnLabelEncoder()), 
    ('one-hot', OneHotEncoder(handle_unknown ='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('categor', cat_transformer, categor_cols)])
```

### Feature Selection
Feeding all of the features to the model at once created too much noise in the data and my scores suffered. I implemented a Recursive Feature Elimination pipeline using Sklearn to extract the best subset of features for the estimators. 

```
from sklearn.feature_selection import RFE

'''
The variables below enable me to test the RFE model for the optimal number of features between 1 and 25
and keeps track of the R2 score and the adjusted R2 score
'''
num_features = np.arange(1,25)
high_score = 0
scores = []
adj_scores = []
high_score_2 = 0

'''
This pipeline preprocesses the data, fits a linear Regression model
and passes it through RFE to determine the optimal features
'''
feature_select = PipelineRFE(
        steps = [
            ('preprocessor', preprocessor),
            ('rfe', RFE(LinearRegression(), num_features[n])),
            ('model', LinearRegression())])

'''
Inside the for loop we are fitting the pipeline and tracking the scores for each iteration
Everytime the score surpasses the high score, I update it so I know what the score was 
when the best subset of features was found
'''
for n in range(len(num_features)):
      
    feature_select.fit(X_train, y_train)
        
    score = feature_select.score(X_test, y_test)
    score_2 = adj_r2_score(feature_select,X_test, y_test)
        
    scores.append(score)
    adj_scores.append(score_2)
        
    if score > high_score:
        high_score = score
        
    if score_2 > high_score_2:
        high_score_2 = score_2
        
temp = pd.Series(feature_select.named_steps['rfe'].support_)
selected_features_rfe = temp[temp==True].index
    
print('Selected Features:{}'.format(selected_features_rfe))          
print('R2 High Score: {}'.format(high_score))
print('Adj R2 High Score: {}'.format(high_score_2))
```
### Model Selection
After determining which columns were best for training the models, I trained each model using those features to compare results. In order to ensure the scores were valid, I used KFold Cross Validation with each model 10 times. 

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
X_train_pre = preprocessor.fit_transform(X_train)
X_test_pre = preprocessor.transform(X_test)
X_train_rfe = X_train_pre[:,selected_features_rfe]

models = [
    LinearRegression(),
    Ridge(),
    Lasso(),
    ElasticNet(),
    GradientBoostingRegressor(),
    RandomForestRegressor()
]

k_fold = KFold(n_splits=10, shuffle=False, random_state=0)

for model in models:
    
    results_kfold = cross_val_score(model, X_train_rfe, y_train, cv=k_fold)
    
    print(model)
    print(np.mean(results_kfold))

```

## Conclusion Part Two
With the selected features, each model scored fairly well aside from the Elastic Net. I chose Lasso Regression because it implements regularization which penalizes irrelevant features. I also chose it based upon the scores that I cross validated, and because it does a good job of dealing with multivariate data while quieting some of the noise caused by low impact features. 
