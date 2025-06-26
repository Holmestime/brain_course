# -*- coding: utf-8 -*-
"""
This module analyzes and visualizes the coefficients of a logistic regression model
trained on brain data features for depression prediction.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import rcParams

def analyze_coefficients():
    """
    Analyze and visualize logistic regression coefficients from model results.

    """
    # Load model performance data
    file_path = "../result/plot_data/M3_interpret.pickle"
    with open(file_path, "rb") as f:
        result = pickle.load(f)

    coefficients = result['coefficients']
    # Calculate mean coefficients across all folds
    mean_coefficients = np.mean(coefficients, axis=0)
    # Sort coefficients in descending order
    mean_coefficients = mean_coefficients[[1, 0, 8, 9, 6, 7, 4, 5, 2, 3, 18, 16, 20, 15, 11, 19, 10, 21, 23, 22, 12, 13, 17, 14]]

    
    # Set up feature names and counts
    feature_count = len(mean_coefficients)
    
    # Create figure with specified dimensions (in cm, converted to inches)
    plt.figure(figsize=(7.2 / 2.54, 5.5 / 2.54))
    
    # Define color scheme for different feature groups
    colors = ["#8ECFC9", "#BEB8DC", "#E7DAD2"]
    
    # Here, the features are divided into three groups: PSD features,PAC features, temporal features
    legend_handles = []
    for i in range(feature_count):
        feature_num = i + 1
        
        # Assign color and position based on feature group
        if 1 <= feature_num <= 10:
            color = colors[0]
            position = i + 1
        elif 11 <= feature_num <= 24:
            color = colors[1]
            position = i + 4
        else:
            color = colors[2]
            position = i + 7
            
        # Plot horizontal bar for each coefficient
        bar = plt.barh(position, mean_coefficients[i], color=color, edgecolor="black")
        legend_handles.append(bar)


    #
    # Set axis labels and ticks
    plt.ylabel("Feature ID")
    tick_positions = [1, 10, 14, 27]
    tick_labels = [1, 10, 11, 24]
    plt.yticks(tick_positions, tick_labels)
    plt.xlabel("Absolute LR coefficients")
    
    # Configure axes appearance
    ax = plt.gca()
    ax.grid(axis='x', linestyle='-', color='black', alpha=0.3)
    ax.tick_params(axis='x', length=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust layout and save figures
    plt.tight_layout()

    png_path = '../result/plot_png'
    plt.savefig(f'{png_path}/M3_interprete.png', dpi=300, format='png')
    plt.show()

if __name__ == "__main__":
    analyze_coefficients()
