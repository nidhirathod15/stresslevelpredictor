# Stress Level Classification Predictor
Developed a robust **multi-class classification model** to accurately predict an individual's stress level (categorized as Low, Medium, or High) based on a dataset comprising 20 psychological and physiological indicators. The project focuses on building a generalized, production-ready solution with optimized performance and clear feature interpretability.

## Methodology

### 1\. Data Preprocessing and Engineering

  * **Data Cleaning:** Confirmed dataset completeness with **zero missing values** across all key features.
  * **Feature Scaling:** Applied **MinMaxScaler** to normalize all numerical features, ensuring all variables contributed equally to the model training process. Crucially, scaling was fit **only on the training data** to strictly prevent data leakage.
  * **Target Variable:** The `stress_level` column was used as the target for classification.

### 2\. Modeling Strategy
A multi-stage modeling strategy was employed to ensure both high accuracy and model stability.

| Stage | Models Tested | Key Technique | Outcome |
| :--- | :--- | :--- | :--- |
| **Baseline Comparison** | Logistic Regression, Decision Tree, Random Forest | 5-Fold Cross-Validation | Identified **Logistic Regression** as the best-performing and most stable baseline classifier. |
| **Optimization** | Logistic Regression | **GridSearchCV** | Tuned hyperparameters to find the optimal regularization strength and penalty, maximizing predictive power. |
| **Interpretation** | Random Forest | Feature Importance Analysis | Quantified the impact of each independent variable on the final prediction. |

-----

## Key Findings and Results

### Model Performance Summary
After tuning, the **Logistic Regression** model delivered exceptional performance, confirmed by both test set evaluation and cross-validation stability.

| Metric | Baseline (LogReg) | **Tuned Model (LogReg)** |
| :--- | :--- | :--- |
| **Test Accuracy** | $93.84\%$ | **$94.25\%$** |
| **CV Accuracy (5-Fold)** | $93.20\%$ | $94.10\%$ |
| **F1-Weighted Score** | $0.938$ | $0.942$ |

### Final Model Selection
The final model chosen was the **Hyperparameter-Tuned Logistic Regression Classifier**. It achieved a highly stable accuracy of **94.25%** on unseen test data and maintained a similar score through 5-Fold Cross-Validation, demonstrating excellent generalization capabilities and readiness for deployment.

### Top Feature Importance
The Random Forest classifier provided critical insight into the key drivers of stress prediction. The **top 5 most influential features** were identified:

| Rank | Feature | Interpretation |
| :--- | :--- | :--- |
| 1 | `anxiety_level` | Strongest positive correlation with stress. |
| 2 | `sleep_quality` | Major factor; poor quality significantly increases prediction of higher stress. |
| 3 | `SelfEsteem` | A key psychological buffer; lower scores predict higher stress. |
| 4 | `concentration_level` | Directly related to cognitive function under stress. |
| 5 | `breathing_rate` | Physiological marker of stress and anxiety. |

## Technical Stack

**Language:** Python
**Core Libraries:** Pandas, NumPy
**Machine Learning:** Scikit-learn (`LogisticRegression`, `RandomForestClassifier`, `GridSearchCV`, `cross_validate`, `MinMaxScaler`)
**Visualization:** Matplotlib, Seaborn
**Deployment Asset:** Final model exported using `joblib`.

## Future Enhancements

  * **Deep Learning:** Explore performance using a simple Neural Network (e.g., Keras/TensorFlow) to see if it can capture non-linear relationships better than the current models.
  * **SHAP Analysis:** Implement SHAP (SHapley Additive exPlanations) values for a more granular, instance-level explanation of feature influence.
  * **Real-time Data Simulation:** Integrate with a simulated data stream to test model latency and prediction performance in a quasi-real-time environment.
