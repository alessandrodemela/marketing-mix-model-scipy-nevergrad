import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd


def perform_eda(data):
    """
    Performs Exploratory Data Analysis (EDA) on the given DataFrame.

    This function prints summary statistics of the DataFrame, generates
    a correlation heatmap, and creates time series plots for each column
    in the DataFrame.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing time series data.

    Returns:
    None
    """
    # Print summary statistics
    print(data.describe())
    print("\nCorrelation plot:")

    # Create a subplot for the correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    plt.title("Correlation Heatmap")
    plt.show()

    # Create subplots for time series of each column
    num_columns = len(data.columns)
    fig, axes = plt.subplots(
        nrows=num_columns,
        ncols=1,
        figsize=(12, 4 * num_columns)
        )

    for i, column in enumerate(data.columns):
        sns.lineplot(data=data, x=data.index, y=column, ax=axes[i])
        axes[i].set_title(f"Historical Trend of {column}")
        axes[i].set_xlabel("Date")
        axes[i].set_ylabel(column)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()


def visualize_marketing_channels_and_sales(data, kpi, marketing_channels):
    """
    Visualizes the relationship between marketing channels and sales over time.
    
    This function creates a series of bar plots for each marketing channel,
    with a line plot overlay showing the sales data. The aim is to understand
    how different marketing channels correlate with sales performance.

    Parameters:
    - data (pd.DataFrame): DataFrame containing the marketing channels and sales data.
    - kpi (str): Name of the target KPI
    - marketing_channels (list): List of marketing channel column names to visualize.
    """
    
    plt.figure(figsize=(15, 5 * len(marketing_channels)))
    
    # Create a subplot for each marketing channel
    n_channels = len(marketing_channels)
    for i, channel in enumerate(marketing_channels, 1):
        ax = plt.subplot(n_channels, 1, i)
        
        # Plot the bars for the marketing channel
        ax.bar(data.index, data[channel], alpha=0.5, label=channel, width=5)
        
        # Overlay the sales line plot
        ax2 = ax.twinx()
        ax2.plot(data.index, data[kpi], color='red', label='Sales')
        
        # Add legends
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        # Set titles and labels
        ax.set_title(f'{channel} vs Sales')
        ax.set_xlabel('Date')
        ax.set_ylabel(channel)
        ax2.set_ylabel('Sales')
        
        # Rotate date labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()


def handle_seasonality(data, column, period=52):
    """
    Decomposes a time series into its trend, seasonal, and residual components.

    This function applies seasonal decomposition to the specified column in the
    DataFrame, and plots the original data alongside its decomposed components.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing time series data.
    column (str): The name of the column to decompose.
    period (int): The number of observations per cycle (default is 52).

    Returns:
    pd.DataFrame: The original DataFrame with added columns for trend,
                  seasonal, and residual components.
    """
    # Decompose the time series
    decomposition = seasonal_decompose(
        data[column], model="additive", period=period, extrapolate_trend=True
    )

    # Add components to the DataFrame
    data[f"{column}_trend"] = decomposition.trend
    data[f"{column}_seasonal"] = decomposition.seasonal
    data[f"{column}_residual"] = decomposition.resid

    # Plot the components
    plt.figure(figsize=(15, 10))

    # Original Data
    plt.subplot(4, 1, 1)
    sns.lineplot(data=data, x=data.index, y=column)
    plt.title(f"Original {column}")
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel(column)

    # Trend Component
    plt.subplot(4, 1, 2)
    sns.lineplot(data=data, x=data.index, y=f"{column}_trend")
    plt.title(f"Trend Component of {column}")
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel(f"{column}_trend")

    # Seasonal Component
    plt.subplot(4, 1, 3)
    sns.lineplot(data=data, x=data.index, y=f"{column}_seasonal")
    plt.title(f"Seasonal Component of {column}")
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel(f"{column}_seasonal")

    # Residual Component
    plt.subplot(4, 1, 4)
    sns.lineplot(data=data, x=data.index, y=f"{column}_residual")
    plt.title(f"Residual Component of {column}")
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel(f"{column}_residual")

    plt.tight_layout()
    plt.show()

    return data


def check_multicollinearity(X):
    """
    Checks for multicollinearity among the features in the input DataFrame using 
    Variance Inflation Factor (VIF).

    Multicollinearity occurs when two or more predictors in a regression model are 
    highly correlated, which can affect the stability and interpretability of the model.

    Parameters:
    X (pd.DataFrame): The input DataFrame containing features for analysis.

    Returns:
    pd.DataFrame: A DataFrame containing features and their corresponding VIF values.
    """
    # Create a DataFrame to store VIF data
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    return vif_data
