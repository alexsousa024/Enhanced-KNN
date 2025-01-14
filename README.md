# Enhanced KNN: Improving Classification with Bagging and Weights
# Overview
This project focuses on enhancing the traditional K-Nearest Neighbors (KNN) algorithm by integrating bagging and weighted neighbors to improve its accuracy and robustness, especially in datasets with overlapping classes. The implementation demonstrates the effectiveness of these modifications through rigorous testing and analysis on multiple datasets.

# Goals
Primary Objective: Improve the accuracy and reliability of the KNN algorithm.
Target Outcome: Enhance predictive performance while ensuring robustness against class overlap and noise.

Enhanced KNN Algorithm:

Bagging: Combats overfitting by training multiple KNN models on bootstrap samples.
Weighted Neighbors: Ensures closer neighbors contribute more to predictions.


Uses metrics like accuracy, precision, recall, F1-score, and ROC-AUC.
Visualizations include ROC curves, confusion matrices, and accuracy graphs.
Data Preprocessing:

Handles feature scaling, normalization, and encoding to prepare datasets for KNN.
Comprehensive Testing:

Tested on multiple datasets from OpenML, focusing on noise handling, class overlap, and feature scaling.

# Project Structure

project/
├── enhanced_KNN_1.ipynb            # Jupyter Notebook with implementation and analysis
├── csv_result-dataset_diabetes_35.csv  # Processed dataset results for the diabetes dataset
├── heart_statlog_cleveland_hungary_final.csv  # Original dataset for heart disease analysis
├── presentation.pdf                # Detailed project presentation
# Methodology
Algorithm Enhancements
Bagging:

Trains multiple KNN models on different subsets of the dataset (bootstrap samples).
Combines predictions through majority voting (classification) or averaging (regression).
Weighted Neighbors:

Each neighbor's influence is weighted inversely to its distance from the query point.
Prevents domination by distant neighbors and focuses on relevant local data.
Experimental Setup
Datasets:

Sourced from OpenML.
Includes features with varying noise levels and class overlaps.
# Evaluation:

70-30 train-test split.
Performance metrics include accuracy, precision, recall, F1-score, and ROC-AUC.
Results
# Improvements:

Enhanced accuracy in noisy and overlapping datasets.
Improved stability through bagging.
Challenges:

In some datasets, accuracy decreased due to mismatched distance metrics or improper weighting schemes.
Key Findings:

Bagging reduced the impact of outliers and improved robustness.
Weighted neighbors enhanced the algorithm's sensitivity to closer, more relevant data points.
# Installation
Prerequisites
Python 3.8+
Jupyter Notebook
Required Libraries: numpy, scikit-learn, matplotlib, seaborn, pandas
