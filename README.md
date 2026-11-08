# dry-bean-ml-classification
Comparative analysis of five Machine Learning classification models using the Dry Bean Dataset with an interactive Streamlit application
# Dry Bean Classification Using Machine Learning

## 1. Problem Statement

The objective of this project is to classify different varieties of dry beans based on their physical and morphological characteristics using supervised machine learning algorithms.

Five classification algorithms are implemented and evaluated using the same dataset. An interactive Streamlit web application is developed to compare the performance of these models using uploaded test data.

## 2. Dataset Description

**Dataset:** Dry Bean Dataset  
**Source:** Kaggle (originally from the UCI Machine Learning Repository)  
**Problem Type:** Multiclass Classification

The dataset contains measurements of seven different varieties of dry beans.

| Property | Description |
|---|---|
| Original instances | 13,611 |
| Input features | 16 |
| Target variable | Class |
| Number of classes | 7 |
| Missing values | 0 |
| Duplicate records removed | 68 |
| Instances after cleaning | 13,543 |
| Training instances | 10,834 |
| Testing instances | 2,709 |

The seven classes are SEKER, BARBUNYA, BOMBAY, CALI, HOROZ, SIRA and DERMASON.

The input features describe the physical properties of beans, including area, perimeter, axis lengths, aspect ratio, eccentricity, convex area, solidity, roundness and compactness.

### Data Preprocessing

1. Loaded the dataset using Pandas.
2. Checked missing values and duplicate records.
3. Removed 68 duplicate rows.
4. Separated the 16 input features from the target variable.
5. Performed an 80:20 stratified train-test split.
6. Applied StandardScaler to the training data and used the fitted scaler to transform test data.
7. Trained and evaluated all five classification models on the same split.

## 3. GitHub Repository Link

https://github.com/aaryajoshi2608/dry-bean-ml-classification

## 4. Machine Learning Models Used

The following five algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

### Model Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9199 | 0.9935 | 0.9205 | 0.9199 | 0.9200 | 0.9032 |
| Decision Tree | 0.8966 | 0.9363 | 0.8965 | 0.8966 | 0.8964 | 0.8750 |
| KNN | 0.9155 | 0.9811 | 0.9163 | 0.9155 | 0.9157 | 0.8978 |
| Naive Bayes | 0.8970 | 0.9899 | 0.8997 | 0.8970 | 0.8972 | 0.8762 |
| Support Vector Machine | **0.9210** | 0.9933 | **0.9214** | **0.9210** | **0.9211** | **0.9045** |

Multiclass AUC was calculated using the one-vs-rest approach with weighted averaging. Precision, Recall and F1 were calculated using weighted averaging. All models were evaluated using the same test dataset.

## 5. Observations on Model Performance

| ML Model | Observation |
|---|---|
| Logistic Regression | Achieved 91.99% accuracy and the highest AUC of 0.9935. It demonstrated strong classification performance and was close to SVM in overall accuracy. |
| Decision Tree | Achieved 89.66% accuracy, the lowest among the five models. Its unrestricted tree structure may have increased the risk of overfitting. |
| KNN | Achieved 91.55% accuracy using five nearest neighbors. Feature scaling helped ensure that distance calculations were not dominated by large-valued features. |
| Naive Bayes | Achieved 89.70% accuracy and an AUC of 0.9899. Its high AUC compared with its lower accuracy suggests good class-ranking ability despite classification errors. |
| Support Vector Machine | Achieved the highest accuracy of 92.10%, F1 score of 0.9211 and MCC of 0.9045. The RBF kernel enabled nonlinear classification boundaries. |

### Overall Winning Model

**Support Vector Machine (SVM)** was the best-performing model based on test accuracy.

SVM achieved 92.10% accuracy, with the highest F1 score and MCC among the five algorithms.

Logistic Regression achieved a slightly higher AUC, indicating that model selection depends on the chosen evaluation metric.

## 6. Streamlit Web Application

The interactive application includes:

- CSV test dataset upload
- Model selection dropdown for all five classifiers
- Accuracy, AUC, Precision, Recall, F1 and MCC
- Confusion matrix visualization
- Prediction results table
- Downloadable prediction results

The application loads previously trained models and preprocessing artifacts without retraining them on uploaded test data.

## 7. Project Files

- `ML_Assignment_2.ipynb` — Data preparation, model training and evaluation
- `app.py` — Streamlit application
- `requirements.txt` — Python dependencies
- `test_data.csv` — Test dataset
- `model/` — Five trained classifiers, fitted scaler and feature names
- `README.md` — Project documentation

## 8. Conclusion

This project compared five machine learning classification models using the Dry Bean Dataset. The Support Vector Machine achieved the highest test accuracy of 92.10%, followed closely by Logistic Regression at 91.99%.

The results demonstrate that different algorithms have varying strengths, and evaluating multiple metrics provides a more complete assessment of classification performance.

The trained models were integrated into a Streamlit application, allowing users to upload test data and compare model predictions and evaluation metrics.
