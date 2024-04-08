"""
    This file opens csv files from folders with the experiment name.
    Each file has the number of a experiment run with the same parameters.
    Each file has one metric per column with the first row being its name and one time step per row.
    All the experiments are processed.
    The files are read and the average and standard deviation are calculated.
    The average and standard deviation are plotted together in a graph.
    Use pandas.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import chi2

if __name__ == '__main__':
    exp_folder = './results/'
    # Exp names are gathered from the folder names
    exp_names = os.listdir(exp_folder)

    dataframe = pd.DataFrame(
        columns=['timestep', 'run_name', 'baseline', 'entropy'])

    # For each experiment name
    for exp_name in exp_names:
        # Get the files inside the folder
        files = os.listdir(exp_folder + exp_name)
        # For each file
        for file in files:
            # Check if it is a csv file and skip if not (folder)
            if not file.endswith('.csv'):
                continue
            # Read the csv file. headers are in the first row
            csv_dataframe = pd.read_csv(exp_folder + exp_name + '/' + file)

            # Get the headers from the csv file
            columns = csv_dataframe.columns
            # Add baseline, run_name to the headers
            columns = ['baseline', 'run_name'] + columns.to_list()
            run_dataframe = pd.DataFrame(columns=columns)

            for run_dataframe_column in run_dataframe.columns:
                if run_dataframe_column in csv_dataframe.columns:
                    run_dataframe[run_dataframe_column] = csv_dataframe[run_dataframe_column]

            # Get the baseline from the folder exp_name and add it to all the rows
            run_dataframe['baseline'] = [
                exp_name] * len(csv_dataframe['entropy'])
            
            # Get the run name from the file name and add it to all the rows
            run_dataframe['run_name'] = [file.split('_')[-1].split(
                '.')[0]] * len(csv_dataframe['entropy'])

            # concatenate the run dataframe to the main dataframe
            dataframe = pd.concat([dataframe, run_dataframe])

    # Compute mean and standard deviation for each experiment
    mean_metrics = dataframe.groupby(['baseline', 'timestep']).mean()
    std_metrics = dataframe.groupby(['baseline', 'timestep']).std()

    # UNCOMENT TO COMPUTE DIFFERENCES
    # 1. Create a DataFrame with Common Columns (excluding 'baseline')
    common_columns = dataframe.columns.to_list()
    common_columns.remove('baseline')
    common_df = dataframe[common_columns]

    # 2. Extract the Two Baselines
    baseline_1 = dataframe[dataframe['baseline'] == 'metric_basic']
    baseline_1 = baseline_1.drop(columns=['baseline'])
    baseline_2 = dataframe[dataframe['baseline'] == 'metric_utility']
    baseline_2 = baseline_2.drop(columns=['baseline'])

    # 3. Align and Compute Differences
    baseline_1.set_index(['timestep', 'run_name'], inplace=True)
    baseline_2.set_index(['timestep', 'run_name'], inplace=True)

    # Calculate the differences for each metric
    baseline_diff = baseline_1 - baseline_2

    # Reset the index to make 'timestep' and 'run_name' regular columns
    baseline_diff.reset_index(inplace=True)

    # Plot the mean and standard deviation
    sns.set(style="darkgrid")

    plot_metrics_names = ['rmse', 'avg_nees']
    for metric_name in plot_metrics_names:
        for baseline in exp_names:
            data_mean = mean_metrics[metric_name][baseline]

            if metric_name == 'avg_nees':
                chi_squared = chi2.ppf(0.95, 2)
                data_mean = data_mean / chi_squared
            plt.plot(data_mean.index, data_mean,
                     label=baseline)

            data_std = std_metrics[metric_name][baseline]
            plt.fill_between(
                data_std.index,
                data_mean - data_std,
                data_mean + data_std,
                alpha=0.2)
        
        # Add figure title
        plt.title(metric_name)
        plt.legend()
        plt.show()

    # UNCOMENT TO PLOT DIFFERENCES
    # Group the data by 'run_name'
    grouped = baseline_diff.groupby('run_name')

    # Iterate through the metrics and create separate plots for each metric
    for metric in plot_metrics_names:
        plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
        plt.title(f'Difference Over Time for Metric: {metric}')
        plt.xlabel('Timestep')
        plt.ylabel(f'Difference in {metric}')

        # Iterate through each run and plot the difference for the current metric
        for run_name, group_data in grouped:
            plt.plot(group_data['timestep'], group_data[f'{metric}'], label=f'Run: {run_name}')

        plt.legend(loc='upper right')  # Adjust the legend location as needed
        plt.grid(True)

        # Save or display the plot
        # plt.savefig(f'{metric}_difference_plot.png')  # To save the plot as an image
        plt.show()  # To display the plot


