import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


def plot_sentiment_heatmap(df, save_path, dpi=600, fig_size=(16, 6)):
    """
    Plot a heatmap of sentiments.

    Parameters:
    - df: A dataframe containing the sentiment data. Must have columns ['Date', 'Country', 'Average'].
    - save_path: String. Path to save the heatmap image.
    - dpi: Integer. DPI for the saved image.
    - fig_size: Tuple. Size of the figure.
    """
    # Create a copy to avoid modifying the original dataframe
    df_sentiment2 = df[['Date', 'Country', 'Average']].copy()
    df_sentiment2['Date'] = pd.to_datetime(df_sentiment2['Date'], utc=True).dt.date
    df_sentiment2 = pd.pivot_table(df_sentiment2, index=["Date"], columns="Country", values="Average")

    ax = sn.heatmap(df_sentiment2, cmap="YlGnBu")
    ax.xaxis.tick_top()
    plt.xticks(rotation=90)

    fig = plt.gcf()
    fig.set_size_inches(fig_size)

    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')


# 使用方法
# df_sentiment = ...  # your sentiment data
# plot_sentiment_heatmap(df_sentiment, '../Data/Images/Heatmap.png')


def check_correlation(data, target_variable='Return'):
    """
    Calculate the correlation of all columns in the data with a specific target variable.

    Parameters:
    - data: DataFrame. The dataset containing all the variables.
    - target_variable: String. The column name of the target variable against which correlations are checked.

    Returns:
    - result: DataFrame. Contains correlations of all columns with the target variable.
    """
    # Build correlation matrix
    correlations = data.corr(numeric_only=True).unstack().sort_values(ascending=False)

    # Convert correlations to dataframe
    correlations = pd.DataFrame(correlations).reset_index()

    # Label the dataframe
    correlations.columns = [target_variable, 'Method', 'Correlation with ' + target_variable]

    # Filter the dataframe based on the target variable and exclude self-correlation
    result = correlations.query(f"{target_variable} == '{target_variable}' & Method != '{target_variable}'")

    # Drop the target variable column for clarity
    result = result.drop([target_variable], axis=1)

    return result


# 使用方法
# merged_data = ...  # your dataset
# correlations_result = check_correlation(merged_data, 'Return')
# print(correlations_result)


def plot_correlation_heatmap(data, columns, title="Correlation Heatmap", figsize=(10, 6), cmap="Blues"):
    """
    Plot a heatmap showing correlations between given columns.

    Parameters:
    - data: DataFrame containing the data.
    - columns: List of columns for which to compute and display correlations.
    - title: Title for the heatmap.
    - figsize: Tuple indicating the size of the plot.
    - cmap: Color map for the heatmap.

    Returns:
    - ax: The Axes object containing the plot.
    """

    # Extract selected columns
    df_selected = pd.DataFrame(data, columns=columns)

    # Compute correlation matrix
    corr_matrix = df_selected.corr()

    # Create a mask for the upper triangle of the matrix
    mask = np.triu(corr_matrix)

    # Plot heatmap
    plt.figure(figsize=figsize)
    ax = sn.heatmap(corr_matrix, annot=True, vmin=0, vmax=1, cmap=cmap, mask=mask)
    plt.title(title, fontsize=17, fontweight="bold")

    return ax


def compute_strategy_performance(merged_data, strategies, all_dates):
    """
    Compute the performance for each strategy on given dates.

    Parameters:
    - merged_data: DataFrame containing the data.
    - strategies: List of strategies for which to compute the performance.
    - all_dates: List of dates for which to compute the performance.

    Returns:
    - dict_list: A list of dictionaries. Each dictionary corresponds to a strategy and contains date-performance pairs.
    """

    dict_list = [{} for _ in range(len(strategies))]

    for x, strategy in enumerate(strategies):
        for date in all_dates:
            sample_df = merged_data[merged_data['Date'] == date]
            sample_df = sample_df.dropna(subset=[strategy])

            # Adding a slight random perturbation to avoid ties
            # sample_df[strategy] = sample_df[strategy].apply(lambda val: val + np.random.uniform(-0.00001, 0.00001))

            large_df = sample_df.nlargest(5, strategy)
            small_df = sample_df.nsmallest(5, strategy)

            total_ret_long = large_df['Return'].sum() / max(1, len(large_df))
            total_ret_short = small_df['Return'].sum() / max(1, len(small_df))
            total_return = total_ret_long - total_ret_short

            dict_list[x][date] = total_return

    return dict_list


# 使用这个函数：
# strategies = ['LMD','HIV4','Vader', 'FinBert', 'Average']
# all_dates = ...  # your list of dates
# dict_ret_list = compute_strategy_performance(merged_data, strategies, all_dates)


def compute_information_ratio(portfolio_data, strategies):
    """
    Compute the Information Ratio for given strategies.

    Parameters:
    - portfolio_data: DataFrame containing the strategies returns.
    - strategies: List of strategies for which to compute the Information Ratio.

    Returns:
    - IR_dict: Dictionary containing Information Ratio for each strategy.
    """

    IR_dict = {}

    for strategy in strategies:
        returns = portfolio_data[strategy]
        average_return = np.mean(returns)
        std_return = np.std(returns)

        IR = average_return / std_return if std_return != 0 else 0  # Avoiding division by zero
        IR_dict[strategy] = IR

    return IR_dict


# 使用这个函数：
# strategies = ['LMD', 'HIV4', 'Vader', 'FinBert', 'Average']
# IR_values = compute_information_ratio(new_portfolio_25, strategies)
# print(IR_values)

def compute_annualized_return(df, strategies):
    """
    Compute the annualized return for each strategy.

    Parameters:
    - df: DataFrame with 'Date' column and columns for each strategy's return.
    - strategies: List of strategies to compute annualized return for.

    Returns:
    - annualized_returns: Dictionary with strategies as keys and annualized returns as values.
    """

    # Determine the total number of trading days in the dataset
    total_days = df['Date'].nunique()

    # Dictionary to store the annualized returns
    annualized_returns = {}

    for strategy in strategies:
        # Compute the cumulative return for the strategy
        total_return = df[strategy].sum()

        # Calculate the annualized return
        annualized_ret = (1 + total_return) ** (252 / total_days) - 1
        annualized_returns[strategy] = annualized_ret

    return annualized_returns


# Example usage:
# annualized_ret = compute_annualized_return(new_portfolio_25, strategies)
# print(annualized_ret)


def plot_portfolio_summary(new_portfolio, start_date='2016-01-01', end_date='2023-06-30',
                           filename='../Data/Images/portfolio.png', dpi=900):
    """
    Plot various summaries for the given portfolio.

    Parameters:
    - new_portfolio: DataFrame containing portfolio returns. Must have 'Date' as a column.
    - start_date: Starting date for the x-axis.
    - end_date: Ending date for the x-axis.
    - filename: Filepath for saving the resulting image.
    - dpi: Dots per inch for the saved image.

    Returns:
    - fig: The Figure object containing the plots.
    """

    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(14, 8))
    plt.subplots_adjust(hspace=1.0)

    new_portfolio = new_portfolio.set_index('Date')
    new_portfolio.plot(ax=ax1, title='Daily Returns of Portfolio', xlabel='Date', ylabel='Daily Returns (%)', grid=True)
    new_portfolio.cumsum().plot(ax=ax2, title='Cumulative Returns of Portfolio', xlabel='Date', ylabel='Cum Ret (%)',
                                grid=True)
    new_portfolio.rolling(30).mean().plot(ax=ax3, title='MA(30) Returns of Portfolio', xlabel='Date',
                                          ylabel='MA Returns (%)', grid=True)
    new_portfolio.rolling(30).std().plot(ax=ax4, title='MSTD(30) Returns of Portfolio', xlabel='Date',
                                         ylabel='MSTD Returns (%)', grid=True)

    # Creating date range
    date_range = pd.date_range(start=start_date, end=end_date)

    # Creating labels (choosing every start of the year)
    labels = date_range[date_range.is_year_start]

    # Setting ticks and labels
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticks(labels)
        ax.set_xticklabels(labels.year)
        ax.tick_params(labelrotation=45)

    # Preventing label overlap
    plt.gcf().autofmt_xdate()

    # Save the figure
    plt.savefig(filename, dpi=dpi)

    return fig

# 使用这个函数：
# plot_portfolio_summary(new_portfolio_25)
# plt.show()
