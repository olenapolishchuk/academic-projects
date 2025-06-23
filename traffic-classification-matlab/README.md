# Road Traffic Classification (MATLAB)

This project explores machine learning methods for classifying road traffic based on signal data. It was developed as part of a university research collaboration.

## Summary

The goal was to classify between **normal traffic** and **conjugation (traffic congestion)** using time-windowed signal features. The project included:

- Data preparation and segmentation
- Feature generation using lag matrices
- Model training and evaluation with:
  - SVM (RBF kernel)
  - Random Forest (with various tree counts)
  - Classification Tree
  - Linear Regression (as baseline)

## My Contribution

This was a collaborative project. My contributions included:
- Full **data preprocessing**: dataset segmentation, lag matrix generation, label preparation.
- Implementation and tuning of:
  - **Support Vector Machine (SVM)**
  - **Random Forest** models

## Results

| Model              | Accuracy |
|-------------------|----------|
| SVM (RBF)         | 98.05%   |
| Random Forest (200 trees) | 98.21%   |
| Classification Tree | 96.42%   |

## Files

- `dataScenario2v3.mat`: Raw input data (signal samples).
- `traffic_classification.mat`: Contains preprocessing and model training script.
- `report.pdf`: Presentation of project results.

## Tools

- MATLAB
- Statistics and Machine Learning Toolbox


