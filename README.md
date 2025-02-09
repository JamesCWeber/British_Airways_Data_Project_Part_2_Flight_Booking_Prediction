# Flight Booking Prediction (British Airways Data Science Project Part 2)
![Introductory Picture](Booking_Prediction.png)
## Introduction
This is Part 2 of a project from the [British Airways Data Science micro-internship](https://www.theforage.com/simulations/british-airways/data-science-yqoz). British Airways is the flag carrier airline of the United Kingdom and is the second largest UK based carrier based on fleet size and passengers. British Airways has merged with Iberia, Spain's flag carrier airline, creating the International Airlines Group.

In this task, I take on the role of a junior data scientist employed at British Airways. British Airways has tasked me with collecting customer feedback and reviewing data from a third party source. I am also tasked with creating a predictive model to help determine which customers will book a flight for the upcoming holidays. **In this part of the project, I will analyze and conduct feature analysis on flight booking data, train and test a Random Forest model to predict which customers will book a flight, analyze the model performance with a confusion matrix, conduct feature importance analysis, and present my insights on a PowerPoint slide.**

## Problem Statement
British Airways is taking a proactive approach towards attracting customers. Rather than hoping that a customer purchases flights for the holidays as they come into the airport, British Airways is using data from past bookings and predictive models to determine how they can sell their flights to prospective customers. **Using their booking data, we will create and modify features so that they can be used in a Random Forest model. We will then train and tewst the model and analyze it's performance in order to understand how predictive the data really was and whether we can feasibly use it to predict the target outcome (customers buying holidays). The results will be summarized in a Power Point Slide which will be used in a board meeting.**

## Skills Demonstrated
* Jupyter Notebook
* Python
* Data Cleaning
* Data Manipulation
* Data Visualization
* Feature Engineering
* Machine Learning
* Random Forest
* Training and Testing Machine Learning Models
* Confusion Matrix
* Feature Importance
* Power Point

## Data Sourcing
This data was provided to me by the British Airways Data Science micro-internship hosted by Forage. A copy of the data is included in this repository under the file name: customer_booking.csv.

## Data Attributes
The data that we are using to train and test our Random Forest model is provided by British Airways. The data contains 50000 data points (rows) and 14 attribtues (columns).
* num_passengers - Number of passengers travelling.
* sales_channel - Sales channel booking was made on.
* trip_type - Trip type (Round Trip, One Way, Circle Trip).
* purchase_lead - Number of days between travel date and booking date.
* length_of_stay - Number of days spent at destination.
* flight_hour- Hour of flight departure.
* flight_day - Day of week of flight departure.
* route - Origin to destination flight route.
* booking_origin - Country from where booking was made.
* wants_extra_baggage - If the customer wanted extra baggage in the booking.
* wants_preferred_seat - If the customer wanted a preferred seat in the booking.
* wants_in_flight_meals - If the customer wanted in-flight meals in the booking.
* flight_duration - Total duration of flight in hours.
* booking_complete - Flag indicating if the customer completed the booking.

## Booking Prediction
To predict whether a customer will book a flight or not, we must analyze, manipulate, and transform raw data into necessary features for machine learning. Then we will train and test a Random Forest model, analyze its performance, and determine which features are most important to customers when booking a flight.

A copy of this project is included in this repository under the file name: James Weber Random Forest British Airway.ipynb.

### 1. Importing Libraries and Data
We must first import libraries which contains the commands we need for feature engineering and to train and test a Random Forest model.
Then we import the data from the customer_booking.csv file into the df dataframe.
```
# Import libraries.

import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt

%matplotlib inline

sns.set(color_codes = True)

# Used to calculate accuracy, precision, recall, and confusion matrix.

from sklearn import metrics

# Used to create training and test data.

from sklearn.model_selection import train_test_split

# Used to create a random forest model.

from sklearn.ensemble import RandomForestClassifier

# Used to create plot trees to visualize each random forest decision tree.

from sklearn.tree import plot_tree

# Use the pd.read_csv() to import a csv file into Jupyter notebook.
# Some values in the csv file contain characters not in UTF-8.
# Use encoding = "ISO-8859-1" to include those characters.

df = pd.read_csv(r'C:/Users/jwebe/OneDrive/Desktop/customer_booking.csv', encoding = "ISO-8859-1")
```
Now that we imported the data into the df dataframe, let's get a better understanding of the dataframe by looking up some information on it.
```
# Use the .info() command to see how many non null values are in each column and the data type of each column.

df.info()
```
![Summary of Information on df Dataframe](Dataframe_Info.png)

The picture above depicts a table that contains information on the df dataframe, including column names, number of non null values in each column, and the data type for each column. **All columns contain 50000 non null entries, so there are no missing values to replace. All columns also contain the proper data types.**

Columns that contain object data type can only contain non numeric (categorical) data. Columns that contain int64 data type can only contain whole number data. Columns that contain float64 data type can only contain decimal number data.

### 2. Feature Engineering
**Feature engineering is the process of selecting, manipulating and transforming raw data into new features (attributes or columns) that can be used in machine learning. Machine learning models are not capable of using categorical (non numeric) data as inputs.** The picuture above shows 5 columns with categorical data: sales_channel, trip_type, flight_day, route, and booking_origin.

**Label encoding is a process of converting categorical data into numeric data by replacing categorical data wih numbers.** The flight_day column contains the days of the week in a non numeric format: Mon, Tue, Wed, Thy, Fri, Sat, and Sun. **Categorical data with a specific order or hierarchy, such as days of the week, is called ordinal data.** Since days of the week have a specific order, we can replace Mon with the value of 1, Tue with the value of 2, and so on.

The code below will replace the days of the week with numbers.
```
# Create a dictionary that converts a day of the week into a number.

mapping = {'Mon': 1, 
           'Tue': 2,
           'Wed': 3, 
           'Thu': 4, 
           'Fri': 5, 
           'Sat': 6, 
           'Sun': 7}

# Use the .map() command to map a dictionary to a column.

df['flight_day'] = df['flight_day'].map(mapping)
```

The sales_channel and trip_type columns contain categorical data that needs to be converted to numeric data. **Sales channel and trip type data are also nominal data. Unlike ordinal data, nominal data do not have an order or hierarchy. Rather than using label encoding, it is better to create dummy variables for sales channel and trip type data.**

With label encoding, we replace categorical data with numbers. By doing so, we imply that certain data are more closely related to other data. For example, Thu (labeled as 4) is more closely related to Wed or Fri (labeled as 3 and 5) and less closely related to Mon or Sun (labeled as 1 and 7). This makes sense if the data has an order or hierarchy (ordinal data).

Nominal data do not have a specific order or hierachy, and label encoding may create relationships between data where there aren't any. One way to convert nominal data is to create dummy variables. **When we create dummy variables, we create a new column for each unique value in a column.** The sales_channel column will create 2 new columns: sales_channel_internet and sales_channel_mobile. The trip_type column will create 3 new columns: trip_type_CircleTrip, trip_type_OneWay, and trip_type_RoundTrip.

**Each dummy variable column will either contain a 1 or 0. A value of 1 indicates that a data point contains that attribute and a value of 0 indicate that a data point does not contain that attribute.** If a customer purchased a round trip flight via the internet, the dummy variable columns trip_type_RoundTrip and sales_channel_internet will contain 1 and the remaining dummy variable columns will contain 0.

The code below will create dummy variables for the sales_channel column.
```
# The sales_channel data needs to be converted into numeric data.
# Since sales_channel do not have a particular order or heirarchy, dummy variables are needed.
# Use the pd.get_dummies() command to create dummy variables for sales_channel.

df_dummy_sales_channel = 'sales_channel_' + df['sales_channel']
df_dummy_sales_channel = pd.get_dummies(df_dummy_sales_channel)

# Use the .astype() command to convert Boolean data type into int data type.

df_dummy_sales_channel = df_dummy_sales_channel.astype(int)

# Replace the original sales_channel column with the 2 dummy variable columns.
# Use the .drop() command to drop the sales_channel column.

df = df.drop(columns = 'sales_channel')

# Use the .insert() command to insert a column from one dataframe into another.

df.insert(1, 'sales_channel_Internet', df_dummy_sales_channel.loc[:, 'sales_channel_Internet'])
df.insert(2, 'sales_channel_Mobile', df_dummy_sales_channel.loc[:, 'sales_channel_Mobile'])
```

The code below will create dummy variables for the travel_type column.
```
# The trip_type data needs to be converted into numeric data.

df_dummy_trip_type = 'trip_type_' + df['trip_type']
df_dummy_trip_type = pd.get_dummies(df_dummy_trip_type)
df_dummy_trip_type = df_dummy_trip_type.astype(int)

# Replace the original trip_type column with the 3 dummy variable columns.

df = df.drop(columns = 'trip_type')

df.insert(3, 'trip_type_CircleTrip', df_dummy_trip_type.loc[:, 'trip_type_CircleTrip'])
df.insert(4, 'trip_type_OneWay', df_dummy_trip_type.loc[:, 'trip_type_OneWay'])
df.insert(5, 'trip_type_RoundTrip', df_dummy_trip_type.loc[:, 'trip_type_RoundTrip'])
```

The booking_origin column contains categorical, nominal data. Let's see how many unique values are in the bookin_origin column.
```
# Use the .unique() command to see the unique values in the booking_origin column.

df['booking_origin'].unique()
```
![List of Unique Values in booking_origin column](Booking_Origins.png)

The picture above is a list of unique values in the booking_origins column. There are 103 unique values, not including the (not set) value. **We cannot create dummy variables for all unique values in the booking_origins column. This will add over 100 new columns to the data which will decrease the machine learning model's efficiency and effectiveness.**

**Rather than creating a dummy variable for all unique values, we will group these values into regions. Then we will create dummy variables for each region.** The uniques values can be grouped into 11 different regions: North America, South America, Europe, Nordic, East Asia, South Asia, South East Asia, Middle East, Africa, Oceania, and Caribbean. Values that are labeled (not set) will be placed in an "Unknown" group.

The code below will replace all booking origin locations with a regional location.
```
# Use the .replace() command to replace the booking origin countries with a region.

df['booking_origin'] = df['booking_origin'].replace(['United States', 'Mexico', 'Canada'], 
                                                     'North America')

df['booking_origin'] = df['booking_origin'].replace(['Brazil', 'Argentina', 'Chile', 'Colombia', 'Paraguay', 'Peru'], 
                                                     'South America')

df['booking_origin'] = df['booking_origin'].replace(['United Kingdom', 'Switzerland', 'Poland', 'Estonia', 'Belgium', 'France', 
                                                     'Hungary', 'Netherlands', 'Germany', 'Bulgaria', 'Spain', 'Czechia', 
                                                     'Austria', 'Slovenia', 'Romania', 'Italy', 'Greece', 'Croatia', 
                                                     'Malta', 'Portugal', 'Slovakia', 'Russia', 'Ireland', 'Ukraine', 
                                                     'Belarus', 'Cyprus', 'Turkey', 'Kazakhstan', 'Czech Republic', 'Gibraltar'], 
                                                     'Europe')

df['booking_origin'] = df['booking_origin'].replace(['Denmark', 'Norway', 'Sweden', 'Finland', 'Svalbard & Jan Mayen'], 
                                                     'Nordic')

df['booking_origin'] = df['booking_origin'].replace(['China', 'Japan', 'South Korea', 'Mongolia', 'Hong Kong', 'Taiwan', 'Macau'], 
                                                     'East Asia')

df['booking_origin'] = df['booking_origin'].replace(['India', 'Pakistan', 'Maldives', 'Nepal', 'Sri Lanka', 'Afghanistan', 'Bangladesh', 'Bhutan'], 
                                                     'South Asia')

df['booking_origin'] = df['booking_origin'].replace(['Brunei', 'Myanmar (Burma)', 'Cambodia', 'Timor-Leste', 'Indonesia', 'Laos', 
                                                     'Malaysia', 'Philippines', 'Singapore', 'Thailand', 'Vietnam'], 
                                                     'South East Asia')

df['booking_origin'] = df['booking_origin'].replace(['Iran', 'Israel', 'Saudi Arabia', 'Lebanon', 'United Arab Emirates', 'Jordan', 'Iraq', 
                                                     'Kuwait', 'Bahrain', 'Oman', 'Qatar', 'Egypt', 'Tunisia'], 
                                                     'Middle East')

df['booking_origin'] = df['booking_origin'].replace(['Algeria', 'Kenya', 'Tanzania', 'South Africa', 'Réunion', 'Mauritius', 'Seychelles', 'Ghana'], 
                                                     'Africa')

df['booking_origin'] = df['booking_origin'].replace(['Australia', 'New Zealand', 'Papua New Guinea', 'Solomon Islands', 'Tonga', 
                                                     'New Caledonia', 'Norfolk Island', 'Guam', 'Vanuatu'], 
                                                     'Oceania')

df['booking_origin'] = df['booking_origin'].replace(['Nicaragua', 'Guatemala', 'Panama'], 
                                                     'Caribbean')

df['booking_origin'] = df['booking_origin'].replace(['(not set)'], 
                                                     'Unknown')
```

Now that we have **reduced the number of unique values in the booking_origin column from 103 unique values to 12 unique values**, we will create dummy variables for each unique value.

The code below will create dummy variables for the booking_origin column.
```
# The trip_type data needs to be converted into numeric data.

df_dummy_booking_origin = 'booking_origin_' + df['booking_origin']
df_dummy_booking_origin = pd.get_dummies(df_dummy_booking_origin)
df_dummy_booking_origin = df_dummy_booking_origin.astype(int)

# Replace the original trip_type column with the 13 dummy variable columns.

df = df.drop(columns = 'booking_origin')

df.insert(10, 'booking_origin_North America', df_dummy_booking_origin.loc[:, 'booking_origin_North America'])
df.insert(11, 'booking_origin_South America', df_dummy_booking_origin.loc[:, 'booking_origin_South America'])
df.insert(12, 'booking_origin_Europe', df_dummy_booking_origin.loc[:, 'booking_origin_Europe'])
df.insert(13, 'booking_origin_Nordic', df_dummy_booking_origin.loc[:, 'booking_origin_Nordic'])
df.insert(14, 'booking_origin_East Asia', df_dummy_booking_origin.loc[:, 'booking_origin_East Asia'])
df.insert(15, 'booking_origin_South Asia', df_dummy_booking_origin.loc[:, 'booking_origin_South Asia'])
df.insert(16, 'booking_origin_South East Asia', df_dummy_booking_origin.loc[:, 'booking_origin_South East Asia'])
df.insert(17, 'booking_origin_Middle East', df_dummy_booking_origin.loc[:, 'booking_origin_Middle East'])
df.insert(18, 'booking_origin_Africa', df_dummy_booking_origin.loc[:, 'booking_origin_Africa'])
df.insert(19, 'booking_origin_Oceania', df_dummy_booking_origin.loc[:, 'booking_origin_Oceania'])
df.insert(20, 'booking_origin_Caribbean', df_dummy_booking_origin.loc[:, 'booking_origin_Caribbean'])
df.insert(21, 'booking_origin_Unknown', df_dummy_booking_origin.loc[:, 'booking_origin_Unknown'])
```

The last column that contains non numeric, categorical data is the route column. Let's see how many unique values are in the route column.
```
# Use the .unique() command to see the unique values in the booking_origin column.

df['route'].unique()
```
![List of Unique Values in routes column](Routes.png)

The picture above is a sample of the unique values in the route column. There are 799 different routes that British Airways take. Similar to the booking_origin column, the route column contains too many unique values to create dummy variables with. However, there are no obvious categories that we can group the routes. **Since booking_origins provide information on customer's location, we will remove the route column to reduce redundancy.**

The code below will drop the route column from the dataframe.
```
# Use the .drop() command to delete a column.

df = df.drop(columns = 'route')
```
Here is a list of attributes our dataframe currently have. Attributes that are created during feature engineering are bolded.
* num_passengers - Number of passengers travelling.
* **sales_channel_Internet - Dummy variable representing customers who used the internet to book a flight.**
* **sales_channel_Mobile - Dummy variable representing customers who used a mobile phone to book a flight.**
* **trip_type_CircleTrip - Dummy variable representing customers who booked a circle trip.**
* **trip_type_OneWay - Dummy variable representing customers who booked a one way trip.**
* **trip_type_RoundTrip - Dummy variable representing customers who booked a round trip.**
* purchase_lead - Number of days between travel date and booking date.
* length_of_stay - Number of days spent at destination.
* flight_hour- Hour of flight departure.
* flight_day - Day of week of flight departure.
* route - Origin to destination flight route.
* **booking_origin_North America - Dummy variable representing a booking origin in the North America region.**
* **booking_origin_South America - Dummy variable representing a booking origin in the South America region.**
* **booking_origin_Europe - Dummy variable representing a booking origin in the Europe region.**
* **booking_origin_Nordic - Dummy variable representing a booking origin in the Nordic region.**
* **booking_origin_East Asia - Dummy variable representing a booking origin in the East Asia region.**
* **booking_origin_South Asia - Dummy variable representing a booking origin in the South Asia region.**
* **booking_origin_South East Asia - Dummy variable representing a booking origin in the South East Asia region.**
* **booking_origin_Middle East - Dummy variable representing a booking origin in the Middle East region.**
* **booking_origin_Africa - Dummy variable representing a booking origin in the Africa region.**
* **booking_origin_Oceania - Dummy variable representing a booking origin in the Oceania region.**
* **booking_origin_Caribbean - Dummy variable representing a booking origin in the Caribbean region.**
* **booking_origin_Unknown - Dummy variable representing a booking origin that was not given.**
* wants_extra_baggage - If the customer wanted extra baggage in the booking.
* wants_preferred_seat - If the customer wanted a preferred seat in the booking.
* wants_in_flight_meals - If the customer wanted in-flight meals in the booking.
* flight_duration - Total duration of flight in hours.
* booking_complete - Flag indicating if the customer completed the booking.

### 3. Data Modeling
Data modeling is the process of converting raw data into insight using algorithms and other systems of equations. **The machine learning model that we will use to predict whether a customer will book a flight is the Random Forest model.** The Random Forest model is an algorithm that combines the output of multiple decision trees to reach a single decision.

#### 3a. Model Sampling
Random Forest model is a supervised learning algorithm. **Supervised learning algorithms requires training data to create the model. Once the model is created using the training data, the model is compared with the test data to determine if the model is overfitting, underfitting, or has good fit.**

The first thing we need to do is to split our dataset into training and test data. Since both training and test data come from the same dataset, they should follow the same pattern even if the data in both sets are different. **By creating a model using the training data and testing the model using the test data, we can determine how well the model can predict the pattern within the data.**

The code below will separate our dependent variabe (booking_complete) from our independent variable (the remaining attributes). The dependent variable will be 
assigned to y and the independent variables will be assigned to X.
```
# Make a copy of our data.
# It's a good idea to keep the original data intact in case you need the data in the original dataframe.
# Use the .copy() command to create a copy of the data.

df = df.copy()

# Separate target variable from independent variables.
# The target variable is the variable you are trying to predict (booking_complete).
# The independant variables are the varibales you will use to predict the target variable (all variables except booking_complete).

y = df['booking_complete']
X = df.drop(columns = ['booking_complete'])
```
We will then split both y and X into training and test data. The test data are randomly selected points from y and X. 25% of the data from y and X will be used for test data. Both y and X contain 50000 data points, so 37500 data points will be training data and 12500 data points will be test data.
```
# Create training and test data.
# The test size the is % of the original data that will be used for test data.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
```

#### 3b. Model Training
To train the model, we need to create a Random Forest model using the training data. **The model will use the training data, create decision trees, and will predict whether a customer will book a flight or not by finding and learning patterns within the training data.**

The code below will take the training data from y and X, and create a Random Forest model. The Random Forest model consist of many decision trees made up of randomly selected data points and attributes (from the training data). Our Random Forest model will create 1000 decision trees.
```
# Use the RandomForestClassifier() command to create a random forest model.

model = RandomForestClassifier(n_estimators = 1000, 
                               random_state = 42)

# Use the .fit() command to fit the data into the model.
# The data that we will fit into the model will be the training data, both x_train and y_train.

fitted = model.fit(X_train, y_train)
```

#### 3c. Model Testing
Now that we trained the Random Forest model, we will use the test data from y and X to test the model. **We test the model by using the test data as inputs for the Random Forest model. We will then use a confusion matrix to determine the accuracy, precision, and recall for the model.**

The code below will take our trained Random Forest model and use the test data from X as its input. Then we will compare what the model's prediction was with the test data from y to determine the accuracy, precision, and recall for the model.
```
# Use the .predict() command to labels of the data values on the basis of the trained model.
# Use the test data (X_test) as input.

predictions = fitted.predict(X_test)

# Use the metrics.confusion() command to compute confusion matrix to evaluate the accuracy of a classification.
# Use the y_test variable as the true values.
# Use predictions as the predicted values.
# Use the .ravel() command to merge multiple arrays (tn, fp, fn, tp) to a single array.

tn, fp, fn, tp = metrics.confusion_matrix(y_test, predictions).ravel()
```
The code below will print out the accuracy results of the test.
```
## Use the print() command to print out the results of the confusion matrix.

print(f"True positives: {tp}")
print(f"False positives: {fp}")
print(f"True negatives: {tn}")
print(f"False negatives: {fn}\n")

print(f"Accuracy: {metrics.accuracy_score(y_test, predictions)}")
print(f"Precision: {metrics.precision_score(y_test, predictions)}")
print(f"Recall (Sensitivity): {metrics.recall_score(y_test, predictions)}")
print(f"F1: {metrics.f1_score(y_test, predictions)}")
```
![Confusion Matrix](Confusion_Matrix.png)

**True Positives (TP) are events where the model predicted a positive value (a customer will book a flight) and the data supports the model's prediction. False Positives (FP) are events where the model predicts a positive value but the data does not support the prediction.**

**Similarly, True Negatives (TN) are events where the model predicts a negative value (a cusotmer will not book a flight) and the data supports the prediction. False Negatives (FN) are events where the model predicts a negative value but the data does noot support the prediction.**

**Accuracy is the overall accuracy of the model.** It is calculated as TP + TN/TP + TN + FP + FN where TP + TN represents all the predictions that are correct and TP + TN + FP + FN represents all of the predictions.

**Precision is the ability of the model to accurately predict positive values.** It is calculated as TP/TP + FP where TP represents  the values that the model correctly predicts will have a positive value and TP + FP represents all the values that the model predicts will have a positive value, regardless of whether it is correct or incorrect.

**Recall is the ability of the model to accurately detect positive values.** It is calculated as TP/TP + FN where TP represents  the values that the model correctly predicts will have a positive value and TP + FN represents the actual number of positive values.

**F1 represents how well the model can detect and accurately predict positive values.** It is calculated as 2 * (Precision * Recall)/(Precision + Recall).

**Our model has a high accuracy score of 85.07%.** If the model predicts that a value will be positive (yes, a customer will book a flight) or negative (no, a customer will not book a flight), there is a 85.07% chance that the prediction is accurate.

**Our model also has a low precision score of 48.3%. While accuracy represents how well the model is able to predict both positive and negative values, precision represents how well the model is able to predict positive values.** The model has predicted 412 positive values and 199 of the predicted values are true. This indicates that if the model predicts that a value is positive, there is a 48.3% chance that prediction is accurate.

**However, our model has a very low recall score of 10.75%. Recall represents how well the model can detect positive values.** There are a total of 1852 positive values and the model was able to accurately predict 199 of them. This means that our model can only detect 10.75% of positive values.

**Our model also has a low F1 score of 17.58%. The F1 score represents how accurate the model is at predicting and detecting positive values.** A low F1 score indicates the model has issues with accurately predicting postive values either because its predictions tend to be wrong or because it frequently misidentifies positive values.

#### 3d. Model Visualiziation
Now that we have created out Random Forest model, let's create visualizations so that we could better understand the model.

