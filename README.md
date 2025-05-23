# PREDICTIVE-ANALYSIS-USING-MACHINE-LEARNING

**COMPANY**: CODTECH IT SOLUTIONS  
**NAME**: ANGARAG SEAL  
**INTERN ID**: CT04DN666  
**DOMAIN**: DATA ANALYTICS  
**DURATION**: 4 WEEKS  
**MENTOR**: NEELA SANTOSH

---

# Predictive Analysis Using Machine Learning on Breast Cancer Dataset

## Project Overview

This project demonstrates how machine learning techniques can be applied to predict breast cancer diagnosis based on clinical features. Using the Breast Cancer Wisconsin dataset, the goal is to develop, train, and evaluate multiple classification models that accurately distinguish between malignant and benign tumors. The project covers the entire predictive analytics pipeline, from data loading and preprocessing to feature selection, model training, and detailed evaluation, including visual performance analysis.

Machine learning has become a critical tool in healthcare for improving diagnostic accuracy and supporting clinical decisions. By automating the classification of tumors using measurable features, it is possible to reduce the time and cost of diagnosis while improving patient outcomes.

## Dataset Description

The Breast Cancer Wisconsin dataset is a well-known benchmark dataset in the machine learning community for binary classification tasks. It consists of 569 samples collected from digitized images of fine needle aspirate (FNA) of breast masses. Each sample is labeled as either malignant (cancerous) or benign (non-cancerous).

The dataset includes 30 continuous numerical features that describe characteristics of the cell nuclei in the tumor tissue, such as radius, texture, perimeter, area, smoothness, compactness, and concavity. These features provide meaningful information that helps distinguish malignant tumors from benign ones, which is critical in cancer diagnosis.

## Data Preprocessing and Feature Selection

Prior to building predictive models, the dataset undergoes essential preprocessing steps to enhance model performance and interpretability. Initially, the dataset is explored to understand feature distributions and detect any potential anomalies.

Feature selection is applied to reduce dimensionality and improve the model's generalization capability by focusing only on the most informative features. This project uses the SelectKBest method with the ANOVA F-value scoring function (`f_classif`) to identify the top 10 most significant features out of the 30 available. This selection reduces noise and computational overhead while maintaining predictive power.

All selected features are then standardized using `StandardScaler` to normalize the feature scales, which helps many machine learning algorithms converge faster and perform better.

## Model Training and Comparison

Three supervised classification algorithms are trained and evaluated on the processed dataset:

- **Logistic Regression:** A simple yet effective linear model commonly used for binary classification problems, estimating the probability that an input belongs to a particular class.

- **Random Forest Classifier:** An ensemble method that constructs multiple decision trees and aggregates their predictions to enhance accuracy and robustness.

- **K-Nearest Neighbors (KNN):** A non-parametric, instance-based learner that classifies new samples based on the labels of their closest neighbors in feature space.

The data is split into training and testing sets in an 80-20 ratio, ensuring that models are evaluated on unseen data. Each model is trained on the training set, and predictions are generated for the test set. Performance metrics such as accuracy, precision, recall, and F1-score are computed for detailed assessment.

## Model Evaluation and Visualization

The best-performing model, selected based on test accuracy, is further analyzed using graphical methods to better understand its strengths and weaknesses:

- **Confusion Matrix:** Provides a detailed breakdown of correct and incorrect classifications, allowing insight into the types of errors the model makes.

- **ROC Curve and AUC:** The Receiver Operating Characteristic curve illustrates the trade-off between true positive rate and false positive rate at different thresholds, with the Area Under the Curve (AUC) serving as a summary measure of the model’s discriminative ability.

These plots are displayed side-by-side for clear and concise visualization of the model’s performance.

## Conclusion

This project successfully demonstrates the practical use of machine learning to support breast cancer diagnosis, emphasizing the importance of feature selection, model comparison, and thorough evaluation. The methodology and code provide a solid foundation for similar classification tasks in healthcare and other domains, highlighting how data-driven approaches can enhance decision-making processes.

---

## How to Run

1. Ensure Python 3.7+ is installed.
2. Install dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
