# Solar-Energy-Prediction-using-LSTM

## Data collection
The dataset used in this project was collected from the NASA website at 12o 00” latitude, 37o 15” longitude. It originally consisted of 14 columns and 186,262 rows with a 1-hour step from 01/01/2001 through 04/01/2022. There were no missing records or outliers in the dataset. The target column for the prediction was global horizontal irradiance, which represents the amount of solar energy received on a horizontal surface.

##	Pre-processing Data
The preprocessing stage involved several steps to prepare the data for modeling:

### Input parameters definitions:
*	All Sky Surface Shortwave Downward Irradiance, `ALLSKY_SFC_SW_DWN` (Wh/m^2) – The total solar irradiance incident (direct plus diffuse) on a horizontal plane at the surface of the earth under all sky conditions. An alternative term for the total solar irradiance is the “Global Horizontal Irradiance” or GHI.
*	Dew/Frost Point at 2 Meters, `T2MDEW` (C) – The dew/frost point temperature at 2 meters above the surface of the earth.
*	Temperature at 2 Meters, `T2M` (C) - The average air (dry bulb) temperature at 2 meters above the surface of the earth.
*	Wind Direction at 10 Meters, `WD10M` (Degrees) - The average of the wind direction at 10 meters above the surface of the earth 
*	Wind Speed at 10 Meters, `WS10M` (m/s) – The average of wind speed at 10 meters above the surface of the earth.
*	Precipitation Corrected, `PRECTOTCORR` (mm/hour) – The bias corrected average of total precipitation at the surface of the earth in water mass (includes water content in snow).
*	Relative Humidity at 2 Meters, `RH2M` (%) – The ratio of actual partial pressure of water vapor to the partial pressure at saturation, expressed in percent.
*	Surface Pressure, `PS` (kPa) – The average of surface pressure at the surface of the earth. 

### Date-time object creation: 
The first step in cleaning data is creating date time object which is used as index of the data. The date and time information was converted into a suitable format for time-series analysis. 

### Exploratory data analysis: 
This step involved visualizing and summarizing the data to identify trends, patterns, and potential issues that could affect the modeling process.

### Fixing missing observations: 
The most difficult and important part of preprocessing is fixing missing observations. Data gaps must be filled using appropriate gap-filling techniques to obtain a complete time series data set. There are various gap-filling techniques for time series data. Among them four imputation methods were selected. These are:
*	`Linear interpolation`: Linear interpolation is a curve-fitting method that uses linear polynomials to construct data points within the discrete range of known data points.
*	`Spline interpolation`: Spline interpolation is a curve-fitting method that uses a piecewise polynomial interpolant to construct data points within a discrete range of known data points.
*	`Forward fill`: imputation by last observation carried forward.
*	`Backward fill`: imputation by next observation carried backward.
#### The last two methods were excluded because of applicability issues for time series and high deviations of the statistical metrics results. Linear interpolation outperformed for most of the experiments analyzing RMSE although the spline interpolation results did not significantly differ from those of linear interpolation. Computationally, linear interpolation is more appropriate because of its simplicity compared to spline interpolation; therefore, we can conclude that linear interpolation is the most appropriate choice of imputation method.

### Removing outliers: 
The next step in cleaning data is detecting and removing outliers which are extreme values that go beyond the normal value. Outliers were detected by analyzing maximum and minimum values for each parameter. The outliers were removed and replaced using linear interpolation.

### Feature engineering: 
To capture the temporal dependencies in the data, a lag of 1-5 hours was created for the target feature. This allowed the model to consider the historical values of the target feature when making predictions.

### Feature selection: 
Recursive Feature Elimination (RFE) with a random forest model was used to identify and select the 10 most important features for the prediction task. This step helped reduce the dimensionality of the dataset and improve the efficiency of the modeling process. Selected features were 'MO', 'HR', 'T2M', 'RH2M', 'SZA', 'ALLSKY_KT', 'lag_5', 'lag_4', 'lag_2', 'lag_1'.

### Normalization: 
The data was normalized using a min-max scaler to scale the values down to a range of 0-1. This step ensured that all features had equal weight in the model and prevented any one feature from dominating the predictions.

### Sequence creation: 
A window size of 24 was used to create sequences of input data for the LSTM model. Each sequence contained 24 consecutive hours of data, allowing the model to learn from the temporal patterns in the data.

### Train-test split: 
The dataset was split into training and testing sets with an 80:20 ratio. This allowed for the evaluation of the model's performance on unseen data.

##	Experiment Design
The experiment design consisted of the following steps:

### Time-series cross-validation:
A cross-validation technique specifically designed for time-series data was used, with 4 splits and a test size of 3 months. This ensured that the model was evaluated on different time periods, providing a more robust assessment of its performance.

### Model design: 
*	`LSTM-based RNN model`: A deep learning model using Long Short-Term Memory (LSTM) cells was implemented. The model architecture included a variable number of layers and neurons, allowing for the exploration of different model complexities [15]. Adam algorithm was used as optimizer, which can find the optimal solution efficiently by flexibly adjusting the learning rate and model building time through learning. 
*	`Support Vector Regression` (SVR): It is regression technique to minimize error using most suitable hyper-plane. Different kernel functions such as linear, RBF, Sigmoid and Polynomial are used and selecting kernel function is significant for regression. Minimum Loss function is used to improve prediction accuracy. C hyper-parameter is regularization, a tunable parameter that gives more weight to minimizing the error. A flexible tube of minimal radius is formed symmetrically around the estimated function, such that the absolute values of errors less than a certain threshold epsilon are ignored both above and below the estimate. For the C hyper-parameter, a value from 0.01 to 100 is selected and for epsilon a value from 0.001 to 10 is selected.

### Parallelization: 
The cross-validation training process was parallelized using the joblib library from scikit-learn, enabling the model to run concurrently on a remote GPU server hosted by Saturn Cloud platform. This significantly reduced the training time and allowed for more efficient experimentation. 

### Hyperparameter optimization: 
Bayesian optimization was used to search for the optimal hyperparameters for the proposed model. It is a probabilistic approach that uses a model of the objective function to guide the search for the optimal hyperparameters. Hyperparameter bounds were set at 10-500 for neurons and 1-5 for layers. Each optimization step was logged to a JSON file to save previously seen points and facilitate the search process. The proposed model was then trained with the selected best parameters on the entire training dataset.

### Evaluation metrics: 
Two evaluation metrics were used to measure the forecasting performance. These metrics are Root Mean Square Error (RMSE) and Coefficient of determination (R2). The RMSE tells us how well a regression model can predict the value of the response variable in absolute terms while R2 tells us how well a model can predict the value of the response variable in percentage terms.
*	`Root Mean Squared Error` (RMSE): RMSE measure is one of the most preferred scale-dependent measures due to its suitability for evaluating various models built using the same dataset. It is metric that tells us how far apart the predicted values are from the observed values in a dataset, on average.
*	`Coefficient of determination` (R2): A metric that tells us the proportion of the variance in the response variable of a regression model that can be explained by the predictor variables.
