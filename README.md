Introduction: Sales forecasting is the lifeblood of any retail business. It empowers decision-makers to optimize inventory, plan marketing strategies, and allocate resources effectively. In this article, we will embark on an in-depth exploration of time series forecasting using Linear Regression. We'll walk through the entire process: from data preparation to model training and evaluation, and finally, we'll visualize the results with insightful diagrams.
Understanding the Data:
Our dataset, sourced from a fictitious store, comprises historical sales data. Our goal is to predict monthly sales for the upcoming year, leveraging past sales patterns.
pythonCopy code
 Importing essential libraries and loading the dataset import os import pandas as pd import numpy as np import matplotlib.pyplot as plt from sklearn.linear_model import LinearRegression from sklearn.preprocessing import MinMaxScaler from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # Load the sales dataset DF = pd.read_csv('Store.csv') DF.head(10) 
Data Preprocessing:
Before diving into model training, we must prepare the data. This includes handling date columns, transforming the data into a time series format, and dealing with any missing values.
pythonCopy code
Dropping unwanted columns and converting date to a proper format Store_Data = DF.drop(['store', 'item'], axis=1) Store_Data['date'] = pd.to_datetime(Store_Data['date']) # Converting the date to monthly periods and aggregating total sales per month Store_Data['date'] = Store_Data['date'].dt.to_period('M') Monthly_Sales = Store_Data.groupby('date').sum().reset_index() # Handling date format and visualizing monthly sales Monthly_Sales['date'] = Monthly_Sales['date'].dt.to_timestamp() plt.figure(figsize=(15, 5)) plt.plot(Monthly_Sales['date'], Monthly_Sales['sales']) plt.xlabel('Date') plt.ylabel('Sales') plt.title('Monthly Sales') plt.show() 
Time Series Difference:
To improve forecast accuracy, we calculate the difference between consecutive months' sales, a technique known as differencing. This helps in making the time series data stationary.
pythonCopy code
Calculating the difference between consecutive months' sales Monthly_Sales['sales_diff'] = Monthly_Sales['sales'].diff() Monthly_Sales = Monthly_Sales.dropna() 
Model Training Data:
To train our Linear Regression model, we need to create a supervised learning dataset. This involves creating lagged features by shifting the sales differences for the past 12 months.
pythonCopy code
 Creating lagged features for the past 12 months supervised_data = Monthly_Sales.drop(['sales', 'date'], axis=1) for i in range(1, 13): col_name = 'month_' + str(i) supervised_data[col_name] = supervised_data['sales_diff'].shift(i) supervised_data = supervised_data.dropna().reset_index(drop=True) 
Splitting the Data:
We split the data into training and testing sets, scaling the features using Min-Max scaling.
pythonCopy code
Splitting the data into training and testing sets and scaling train_data = supervised_data[:-12] test_data = supervised_data[-12:] Scaler = MinMaxScaler(feature_range=(-1, 1)) Scaler.fit(train_data) train_data = Scaler.transform(train_data) test_data = Scaler.transform(test_data) x_train, y_train = train_data[:, 1:], train_data[:, 0:1] x_test, y_test = test_data[:, 1:], test_data[:, 0:1] y_train = y_train.ravel() y_test = y_test.ravel() 
Model Training and Prediction:
We train our Linear Regression model and make predictions for the next 12 months.
pythonCopy code
 Training the Linear Regression model and making predictions lr_model = LinearRegression() lr_model.fit(x_train, y_train) lr_pre = lr_model.predict(x_test) lr_pre = lr_pre.reshape(-1, 1) lr_pre_test_set = np.concatenate([lr_pre, x_test], axis=1) lr_pre_test_set = Scaler.inverse_transform(lr_pre_test_set) 
Combining Predictions with Actual Sales:
To visualize the predictions alongside actual sales data, we combine them in a new DataFrame.
pythonCopy code
 Combining predictions with actual sales data result_list = [] for i in range(0, len(lr_pre_test_set)): result_list.append(lr_pre_test_set[i][0] + act_sales[i]) lr_pre_series = pd.Series(result_list, name="Linear Prediction") predict_df = predict_df.merge(lr_pre_series, left_index=True, right_index=True) 
Model Evaluation:
We assess the model's performance using key metrics: Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R2).
pythonCopy code
 Model evaluation lr_mse = np.sqrt(mean_squared_error(predict_df['Linear Prediction'], Monthly_Sales['sales'][-12:])) lr_mae = mean_absolute_error(predict_df['Linear Prediction'], Monthly_Sales['sales'][-12:]) lr_r2 = r2_score(predict_df['Linear Prediction'], Monthly_Sales['sales'][-12:]) 
Visualizing Predictions:
Finally, we visualize the predictions and actual sales data over time, providing a clear view of our model's performance.
pythonCopy code
Visualization of predictions plt.figure(figsize=(15, 5)) plt.plot(Monthly_Sales['date'], Monthly_Sales['sales']) plt.plot(predict_df['date'], predict_df['Linear Prediction']) plt.title("Final Predictions") plt.xlabel('Dates') plt.ylabel('Sales') plt.legend(['Actual Sales', 'Predicted Sales']) plt.show() 
Conclusion:
In this comprehensive exploration, we've uncovered the intricacies of time series forecasting with Linear Regression. From data preparation to model evaluation and visualization, we've covered every step of the process.
Accurate sales predictions can be a game-changer for businesses, enabling them to adapt and thrive in a competitive market. As you delve deeper into the world of time series forecasting, remember that Linear Regression is just one tool in your arsenal. More advanced techniques like ARIMA, Prophet, or machine learning models await your exploration, ready to tackle even more complex forecasting challenges.
