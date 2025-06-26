# -*- coding: utf-8 -*-
"""
This module implements machine learning classification models for 
depression prediction based on brain data features.
"""

import numpy as np
import pickle
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from scipy import stats


def load_all_feature_data():
    """
    Load features and labels from the machine learning dataset.
    
    Returns:
        tuple: (features, labels) where features is a matrix of shape (n_samples, n_features)
              and labels is a vector where 1 represents depression and 0 represents normal.
    """
    feature_file = "../dataset/feature/machine_learning_data.mat"
    file_handle = loadmat(feature_file)
    features = file_handle["features"]
    labels = file_handle["labels"]
    # Labels: 1 - depression, 0 - normal
    return features, labels


def calculate_sensitivity_specificity(y_true, y_pred):
    """
    Calculate sensitivity and specificity from confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        tuple: (sensitivity, specificity)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = 0
    specificity = 0
    return sensitivity, specificity


def evaluate_model_performance(data, labels, n_iterations=10, n_splits=2):
    """
    Evaluate the logistic regression model performance using cross-validation.
    
    Args:
        data (ndarray): Feature matrix of shape (n_samples, n_features)
        labels (ndarray): Target labels of shape (n_samples,)
        n_iterations (int, optional): Number of iterations for model evaluation. Defaults to 200.
        n_splits (int, optional): Number of folds for cross-validation. Defaults to 5.
        
    Returns:
        None: Results are saved to a pickle file and plots are generated
    """
    import matplotlib.pyplot as plt
    
    # Initialize lists to store performance metrics
    accuracy_scores = []
    f1_scores = []
    auc_scores = []
    sensitivity_scores = []
    specificity_scores = []
    coefficients = []
    all_y_true = []
    all_y_prob = []
    
    # Run multiple iterations to get stable performance estimates
    for iteration in range(n_iterations):
        print(f"Iteration {iteration+1}/{n_iterations}")
        
        # Create k-fold cross-validation object
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
        
        # Perform cross-validation
        for train_idx, val_idx in kfold.split(data, labels):
            # Split data into training and validation sets
            train_data = data[train_idx]
            train_labels = labels[train_idx]
            val_data = data[val_idx]
            val_labels = labels[val_idx]
            
            # Standardize features
            scaler = StandardScaler()
            train_data_scaled = scaler.fit_transform(train_data)
            
            # Train logistic regression model
            clf = LogisticRegression(class_weight='balanced')
            clf.fit(train_data_scaled, train_labels)
            
            # Standardize validation data using training data statistics
            val_data_scaled = scaler.transform(val_data)
            
            # Generate predictions
            predictions = clf.predict(val_data_scaled)
            probabilities = clf.predict_proba(val_data_scaled)
            
            # Calculate performance metrics
            accuracy_scores.append(accuracy_score(val_labels, predictions))
            f1_scores.append(f1_score(val_labels, predictions))
            auc_scores.append(roc_auc_score(val_labels, probabilities[:, 1]))
            
            # Calculate sensitivity and specificity
            sensitivity, specificity = calculate_sensitivity_specificity(val_labels, predictions)
            sensitivity_scores.append(sensitivity)
            specificity_scores.append(specificity)
            
            coefficients.append(np.squeeze(clf.coef_))
            
            # Store for ROC curve plotting
            all_y_true.extend(val_labels)
            all_y_prob.extend(probabilities[:, 1])

    # Convert lists to arrays for easier analysis
    accuracy_scores = np.array(accuracy_scores)
    f1_scores = np.array(f1_scores)
    auc_scores = np.array(auc_scores)
    sensitivity_scores = np.array(sensitivity_scores)
    specificity_scores = np.array(specificity_scores)
    coefficients = np.array(coefficients)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Bar chart of performance metrics
    metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'F1-Score', 'AUC']
    means = [
        np.mean(accuracy_scores),
        np.mean(sensitivity_scores),
        np.mean(specificity_scores),
        np.mean(f1_scores),
        np.mean(auc_scores)
    ]
    sems = [
        stats.sem(accuracy_scores),
        stats.sem(sensitivity_scores),
        stats.sem(specificity_scores),
        stats.sem(f1_scores),
        stats.sem(auc_scores)
    ]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = ax1.bar(metrics, means, yerr=sems, capsize=5, color=colors, alpha=0.7)
    ax1.set_ylabel('Score')
    ax1.set_title('Model Performance Metrics')
    ax1.set_ylim(0.5, 1)
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Plot 2: ROC curve
    fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
    mean_auc = np.mean(auc_scores)
    sem_auc = stats.sem(auc_scores)
    ax2.plot(fpr, tpr, color='blue', lw=2, 
             label=f'ROC Curve (AUC = {mean_auc:.3f}±{sem_auc:.3f})')
    ax2.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.8)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('../result/plot_png/M2_classification_performance_results.png', dpi=300)
    plt.close()
    
    # Save results to file
    results = {
        'accuracy_scores': accuracy_scores,
        'sensitivity_scores': sensitivity_scores,
        'specificity_scores': specificity_scores,
        'f1_scores': f1_scores,
        'auc_scores': auc_scores,
        'coefficients': coefficients,
        'all_y_true': all_y_true,
        'all_y_prob': all_y_prob
    }
    output_path = "../result/plot_data/M3_interpret.pickle"
    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    
    # Print summary statistics
    print(f"Results saved to {output_path}")
    print(f"Plot saved to ../result/plot_png/M2_classification_performance_results.png")
    print("\n=== Performance Summary ===")
    print(f"Accuracy:    {np.mean(accuracy_scores):.4f} ± {stats.sem(accuracy_scores):.4f}")
    print(f"Sensitivity: {np.mean(sensitivity_scores):.4f} ± {stats.sem(sensitivity_scores):.4f}")
    print(f"Specificity: {np.mean(specificity_scores):.4f} ± {stats.sem(specificity_scores):.4f}")
    print(f"F1-Score:    {np.mean(f1_scores):.4f} ± {stats.sem(f1_scores):.4f}")
    print(f"AUC:         {np.mean(auc_scores):.4f} ± {stats.sem(auc_scores):.4f}")


if __name__ == '__main__':
    # Load data and process
    features, labels = load_all_feature_data()
    labels = labels.ravel()  # Flatten labels array
    
    # Run model evaluation
    # here, we only run 10 iterations for quick testing
    evaluate_model_performance(features, labels, n_iterations=10)
