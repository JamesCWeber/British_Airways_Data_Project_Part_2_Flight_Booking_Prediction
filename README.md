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

## Feature Engineering
**Feature engineering is the process of selecting, manipulating and transforming raw data into new features (attributes or columns) that can be used in supervised machine learning. The features we have in our dataset are listed in the Data Attribures section above.**

**Supervised machine learning is the creation of data models by using labeled datasets to train a model to predict outcomes.**

A copy of this feature engineering project is included in this repository under the file name: James Weber Random Forest British Airway.ipynb.

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

### 2. 



