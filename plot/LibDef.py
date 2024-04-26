import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose


def pairplot_multivariate_density(data):
    """
    Function to create a pair plot for multivariate density series.

    Parameters:
    - Data: DataFrame or array-like, shape (n_samples, n_features)
            The input data series. Each column represents a variable.

    Returns:
    - None (plots the original time series with outliers marked)
    """
    # Create pair plot
    sns.pairplot(data)
    plt.show()


def IsolationForest(data, contamination=0.003, random_state=42):
    data = np.array(data).reshape(-1, 1)
    # Initialize the IsolationForest model
    clf = IsolationForest(contamination=contamination, random_state=random_state)

    # Fit the model and predict outliers
    outliers = clf.fit_predict(data)

    # Extract the outliers from the scaled data
    outliers_data = data[outliers == -1]
    outlier_indices = np.where(outliers == -1)[0]

    outliers_df = pd.DataFrame({'value': outliers_data.flatten()}, index = outlier_indices)
    
    # Plot original data with outliers highlighted
    plt.figure(figsize=(10, 8))
    plt.plot(range(len(data)), data, c='blue', label='Normal')
    plt.scatter(outlier_indices, outliers_data, c='red', label='Outlier')
    plt.title('Isolation Forest Outlier Detection')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


def DBScan(data, window_size=10, eps=0.5, min_samples=5):
    """
    Detect outliers in time series data using DBSCAN.

    Parameters:
    - data: pandas Series or DataFrame, shape (n_samples,)
            Time series data.
    - window_size: int, optional (default=10)
            Size of the sliding window used to create feature vectors.
    - eps: float, optional (default=0.5)
            The maximum distance between two samples for them to be considered as in the same neighborhood.
    - min_samples: int, optional (default=5)
            The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.

    Returns:
    - None (plots the original time series with outliers marked)
    """
    # Create feature vectors using a sliding window approach
    feature_vectors = np.vstack([data[i:i+window_size] for i in range(len(data)-window_size)])
    
    # Normalize the feature vectors
    scaler = StandardScaler()
    feature_vectors_scaled = scaler.fit_transform(feature_vectors)
    
    # Initialize and fit DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(feature_vectors_scaled)
    
    # Identify outliers (data points labeled as -1 by DBSCAN)
    outliers_indices = np.where(dbscan.labels_ == -1)[0]
    
    # Plot the original time series data with outliers marked
    plt.figure(figsize=(10, 8))
    plt.plot(data, label='Time Series Data')
    plt.scatter(outliers_indices, data.iloc[outliers_indices], c='red', label='Outliers')
    plt.ylabel('Value')
    plt.title('DBSCAN Outlier Detection on Time Series Data')
    plt.legend()
    plt.grid(True)
    plt.show()


def IQR(data, threshold=1.5):
    """
    Detect outliers in time series data using the Interquartile Range (IQR) method.

    Parameters:
    - data: pandas Series, shape (n_samples,)
            Time series data.
    - threshold: float, optional (default=1.5)
            The threshold to determine outliers. Data points outside the range
            (Q1 - threshold * IQR, Q3 + threshold * IQR) are considered outliers.

    Returns:
    - DataFrame containing the outliers.
    """
    # Calculate the first and third quartiles
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)

    # Calculate the interquartile range (IQR)
    iqr = q3 - q1

    # Calculate the lower and upper bounds for outliers
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    # Find the outliers
    outliers = data[(data < lower_bound) | (data > upper_bound)]

    # Create DataFrame to hold outliers
    df = pd.DataFrame({'Outlier': outliers})
    

    # Detect outliers using IQR method
    outliers_df = IQR(data['close'])

    # Plot the original time series data with outliers marked
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['close'], color='b', label='Original Data')
    plt.scatter(outliers_df.index, outliers_df['Outlier'], color='r', marker='x', label='Outliers')
    plt.ylabel('Value')
    plt.title('Time Series Data with Outliers Detected (IQR Method)')
    plt.legend()
    plt.grid(True)
    plt.show()


def MAD(data, threshold=3.5):
    """
    Detect outliers in time series data using the Median Absolute Deviation (MAD) method.

    Parameters:
    - data: pandas Series, shape (n_samples,)
            Time series data.
    - threshold: float, optional (default=3.5)
            The threshold to determine outliers. Data points with absolute deviations from the median
            greater than threshold * MAD are considered outliers.

    Returns:
    - DataFrame containing the outliers and their corresponding dates.
    """
    # Calculate the median
    median = np.median(data)

    # Calculate the Median Absolute Deviation (MAD)
    mad = stats.median_abs_deviation(data)

    # Calculate the threshold
    threshold_value = threshold * mad

    # Find the outliers
    outliers = []
    outlier_dates = []
    for idx, val in enumerate(data):
        if val > median + threshold_value or val < median - threshold_value:
            outliers.append(val)
            outlier_dates.append(data.index[idx])

    # Create DataFrame to hold outliers and their dates
    df = pd.DataFrame({'Outlier': outliers, 'Outlier_Date': outlier_dates})

    # Detect outliers using MAD method
    outliers_df = MAD(data['close'])

    # Plot the original time series data with outliers marked
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['close'], color='b', label='Original Data')
    plt.scatter(outliers_df['Outlier_Date'], outliers_df['Outlier'], color='r', marker='x', label='Outliers')
    plt.ylabel('Value')
    plt.title('Time Series Data with Outliers Detected (MAD Method)')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_seasonal_decomposition(data, model={'additive', 'multiplicative'}, figsize=(10, 8)):
    """
    Plot the seasonal decomposition of a time series.

    Parameters:
    - data: pandas Series, shape (n_samples,)
            Time series data.
    - model: {'additive', 'multiplicative'}, optional (default='additive')
            The seasonal decomposition model to use.
    - figsize: tuple, optional (default=(10, 8))
            The size of the figure for plotting.

    Returns:
    - None (plots the seasonal decomposition components).
    """
    # Perform seasonal decomposition
    result = seasonal_decompose(data, model=model)

    # Plot the decomposition components
    plt.figure(figsize=figsize)

    plt.subplot(411)
    plt.plot(result.observed, label='Original')
    plt.legend(loc='best')

    plt.subplot(412)
    plt.plot(result.trend, label='Trend')
    plt.legend(loc='best')

    plt.subplot(413)
    plt.plot(result.seasonal, label='Seasonal')
    plt.legend(loc='best')

    plt.subplot(414)
    plt.plot(result.resid, label='Residual')
    plt.legend(loc='best')

    plt.tight_layout()
    plt.show()