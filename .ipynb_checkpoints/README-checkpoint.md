# Some Effective Anomaly Detection Algorithm

## **Algorithms**

Currently there are **four** flavours implemented here:

 * **ThresholdingAnomalyDetector**: Uses a moving average and standard deviation filter to estimate the expected value ranges of the time series at each point in time. If a data point falls outside a certain number of standard deviations from the moving average, it is considered an anomaly.
 
 * **EMAAnomalyDetector**: This is the same as above but has an additional paramater, `alpha`, which determines the degree of weighting for recent data points compared to older data points. More detail later on.
 
 * **ARIMAAnomalyDetector**: This model works differently from those above, and has an additional parameter, `order`, which determines the Auto-Regressive, Integration, and Moving Average fit for an ARIMA model. The model then predicts the values of the time series dataset. The residuals are calculated by subtracting the predicted values from the actual values. Residuals represent the error in prediction. The standard deviation of the residuals is then computed. The algorithm identifies an observation as an anomaly if the absolute value of its residual is greater than a certain threshold times the standard deviation of residuals. 
 
 * **RealTimeAnomalyDetector**: This implementation uses only the data available at the time of detection, therefore produces different results that the algorithms above. Uses loginc similar to ThresholdingAnomalyDetector.
 
 
## **Important Assumptions**
 
* **Stationarity**: The data is assumed to be stationary, meaning that the statistical properties such as mean and standard deviation are constant over time. This is because the moving average and standard deviation are calculated over a sliding window and used to detect anomalies. If the underlying data is non-stationary, the algorithm may have trouble adapting to changes in the mean and standard deviation over time.

* **Normality**: The algorithm implicitly assumes that the data follows a Gaussian or normal distribution. This is due to the usage of the standard deviation for establishing the threshold. In a normal distribution, about 68% of values lie within one standard deviation of the mean, 95% lie within two, and 99.7% lie within three. If the data is not normally distributed, this percentage can vary, which might affect the performance of the anomaly detection.

* **Independence**: The data points are assumed to be independent of each other. The algorithm does not account for any potential correlation between different data points. If the data points are not independent, the presence of an anomaly could be correlated with the presence of other anomalies, which this algorithm would not detect.


 
## Advantages:

* The algorithms are simple to implement and understand.
* It is effective in detecting anomalies that are significantly different from normal behavior.
* It does not require a labeled dataset for training, making it suitable for unsupervised anomaly detection tasks.
* It can detect anomalies in real-time, making it useful in applications where prompt detection is important.
* It can be applied to various types of time-series data, such as stock prices, server logs, and sensor data.


## Limitations:

* It may not be effective in detecting subtle anomalies that are close to the boundary of normal behavior.
* The threshold parameter is subjective and may require some trial-and-error to find the optimal value.
* It assumes that the time-series data is stationary, i.e., its statistical properties do not change over time.
* It may produce false positives if the data has a high level of noise or if there are sudden changes in the data that are not anomalies.
* It may not be suitable for data that has a large number of missing values or outliers, good preprocessing could help here.

![Interactive Plot](images/interactive_plt.png)
