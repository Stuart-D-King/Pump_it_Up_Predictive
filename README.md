# Pump it up: Data mining the water table in Tanzania

Data used in this repository comes from [_DrivenData_](https://www.drivendata.org/), an online web platform for data science practice competitions aimed at tackling social challenges. The datasets used are a compilation of data from [_Taarifa_](http://taarifa.org/) and the [_Tanzanian Ministry of Water_](http://maji.go.tz/). For this particular challenge, competitors are tasked with creating a statistical model that will predict which water pumps in Tanzania are functional, which are in need of repairs, and which don't work at all.

## Building the model

#### Feature selection
My first step was to read in the data, make feature selection, and clean the resulting dataframe. To select the features used in the model, I first identified and removed redundant features (e.g. __extraction_type__ and __extraction_type_group__ hold the same information). I also removed features deemed to hold little predictive power (i.e. __recorded_by__), or features with a majority of missing values (e.g. __amount_tsh__ and __population__). Once my set of features was determined, I then encoded all categorical variables and filled missing values with either the mode for categorical variables, or the mean for continuous variables. Finally, I created a __pump_age__ feature using the date the data was collected (2013) and the __construction_year__ for each pump.

#### Cross validation
The cleaned data was then split into training and test sets, and the independent variables (X values) for each set were scaled. Out-of-the-box classifiers from SciKit Learn were then implemented using the training data. Classifiers evaluated include Random Forest, K-Nearest Neighbors, XG Boost, Gradient Boost, Ada Boost, and Logistic Regression. All models were scored by taking the mean accuracy, precision, recall, and F1 scores computed from 5-fold cross validation.

Because the **Random Forest** classifier produced slightly better scoring metrics, I ran a hyper parameter grid search for this classifier. The gird search results were incorporated into my final model.

Before settling on the Random Forest classifier, I also constructed a simple multi-layer perceptron neural network using Keras to try and improve prediction accuracy. I performed a grid search on my neural network, but even the network using the best parameters did not predict water pump functional status as well as the Random Forest classifier.

## Model training

The model can be trained by running the following code from the command line:

```
python taarifa_train.py
```

This will create train and test datasets and save them to the __data__ directory (will be created if it does not already exist).

## Make predictions

To make predictions run the following code from the command line:

```
python taarifa_predict.py
```

This will return an accuracy score for the model, as well as an array of predictions for each data observation. The predictions will also be saved to a __predict_results.csv__ in the root directory.

## Additional insights / Next steps
During my analysis, I examined the importance of my model's features. Using the leave-one-out approach, the features with the greatest importance were __gps_height__, __quantity_group__, and __pump_age__. Understanding this, additional analysis into how these features impact the likelihood of pump functionality would be an important next step.
* Bar charts of leave-one-out feature importance and leave-one-out OOB scores can be generated using `taarifa_main.py`

Related, I performed principle component analysis (PCA) on my selected features and plotted the amount of variance explained by each principle component. Based on a visual inspection of the scree plot, further dimensionality reduction using PCA would not account for a majority of the variance in my model. A scree plot can be generated using `taarifa_main.py`.

Finally, my model does not explore interactions between features, which would be useful to increase the interpretability of the model.
