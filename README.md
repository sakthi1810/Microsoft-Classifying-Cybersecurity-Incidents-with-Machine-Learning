#### Microsoft-Classifying-Cybersecurity-Incidents-with-Machine-Learning


### Overview

This project focuses on enhancing Security Operation Centers (SOCs) by developing a machine learning model to classify cybersecurity incidents accurately. Using the GUIDE dataset, the model predicts whether an incident is a True Positive (TP), Benign Positive (BP), or False Positive (FP), assisting SOC analysts in prioritizing threats effectively.

### Skills Acquired

Data Preprocessing and Feature Engineering

Machine Learning Classification Techniques

Model Evaluation Metrics (Macro-F1 Score, Precision, Recall)

Cybersecurity Concepts (MITRE ATT&CK Framework)

Handling Imbalanced Datasets

Model Benchmarking and Optimization

### Domain

Cybersecurity and Machine Learning

### Problem Statement

Security Operation Centers deal with large volumes of cybersecurity incidents. Manually triaging these incidents is time-consuming and prone to human error. The goal of this project is to build a machine learning model that automates this triage process, ensuring accurate classification of incidents to improve response times and overall security posture.

### Business Use Cases

Automated Incident Triage: Reducing analyst workload by automatically classifying incidents.

Incident Response Optimization: Assisting guided response systems with accurate classifications.

Threat Intelligence Enhancement: Leveraging historical data to improve detection capabilities.

Enterprise Security Management: Reducing false positives to ensure analysts focus on real threats.

### Approach

Data Exploration & Preprocessing: Understanding the dataset, handling missing values, feature engineering, and encoding categorical variables.

Data Splitting: Stratified splitting of data into training and validation sets to handle class imbalance effectively.

### Model Training & Selection:

Baseline models (Logistic Regression, Decision Tree) for benchmarking.

Advanced models (Random Forest, Gradient Boosting, XGBoost, LightGBM) for optimization.

### Model Evaluation & Tuning:

Evaluated using Macro-F1 score, Precision, and Recall.

Addressed class imbalance using SMOTE and class weighting.

Hyperparameter tuning for optimal performance.

### Model Interpretation:

Feature importance analysis using SHAP values.

Error analysis to refine misclassified cases.

### Final Evaluation:

Testing on unseen data and comparison with baseline models.

Selecting the best model for deployment.

### Results

Best Model: XGBoost, with a high Macro-F1 score and balanced performance.

Class Imbalance Handling: SMOTE was applied to improve classification of minority classes.

Computational Efficiency: Only 20% of training data was used for efficiency without compromising performance.

### Evaluation Metrics

Macro-F1 Score: Ensures balanced performance across all classes.

Precision: Reduces false positives, improving alert accuracy.

Recall: Ensures actual threats are detected correctly.

### Dataset Overview

The dataset is structured in three levels:

Evidence: Raw data related to security alerts.

Alerts: Consolidation of evidence to detect potential threats.

Incidents: Grouping alerts into security incidents.
The model is trained on 1M triage-annotated incidents with 45 features.

### Technologies Used

Python (Pandas, NumPy, Scikit-Learn, XGBoost, LightGBM)

Data Visualization (Matplotlib, Seaborn)

Machine Learning Model Evaluation Techniques

### Future Enhancements

Exploring deep learning models for improved classification.

Deploying the model as an API for real-time cybersecurity incident triage.

Integrating threat intelligence feeds to enhance prediction accuracy.

### Contributors

Sakthi – Data Science 

### License

This project is open-source and available under the MIT License.
