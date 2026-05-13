
# Data Mining Project: High-Dimensional Classification & Pipeline

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Score](https://img.shields.io/badge/Final--Grade-19.5%20%2F%2020-orange)

## 📌 Overview
This project implements a comprehensive data science pipeline including **missing value imputation**, **anomaly detection**, and **supervised classification**. Designed to handle high-dimensional datasets (1600 rows, 106 columns), the project ensures a clean separation between model development and inference.

**Key Achievement:** Achieved a final model **Test F1-Score of 0.78481**.

### Core Components:
*   `train.py`: Handles the full data pipeline, including preprocessing, anomaly detection, and hyperparameter tuning.
*   `predict.py`: An inference script designed to load the pre-trained model and generate predictions on unseen test datasets.

---

## 🛠 Methodology & Final Choices
The following methods were selected based on extensive cross-validation and performance analysis:

### 1. Pre-processing Pipeline
*   **Missing Value Imputation**
    *   **Numerical Features**: Median imputation to remain robust against outliers.
    *   **Nominal Features**: Mode imputation to preserve categorical frequency.
*   **Anomaly Detection**: **Isolation Forest** was utilized to prune the training data. An effective boundary of **160 outliers** was identified and removed.
*   **Normalization**: **MinMaxScaler** was applied to rescale features to a range of [0, 1].

### 2. Classification Model
The **K-Nearest Neighbors (KNN)** classifier was selected as the optimal model.

**Optimal Hyperparameters:**
*   **Isolation Forest**: `n_estimators=10`, `contamination=0.1`
*   **KNN**: `n_neighbors=9`, `weights='uniform'`, `metric='euclidean'`

---

## 💻 Environment & Requirements
*   **OS**: macOS Sequoia
*   **Language**: Python 3.12
*   **Libraries**: `pandas`, `numpy`, `scikit-learn`

### Installation
```bash
pip install pandas numpy scikit-learn
```
---

## 🚀 Reproducing the Results

1. **Prepare the Data**
Place `DM_Project_24.csv` and `test_data.csv` in the root directory.
2. **Run Training**
```bash
python3 train.py
```
*This preprocesses the data and saves the cleaned training dataset.*

3. **Run Inference**
```bash
python3 predict.py
```
*Loads the pre-trained model to generate predictions on the test set.*

4. **Review Results**
    Final predictions and metrics are exported to: `test_data_prediction.csv`.

---

## 🔬 Justification & Academic Context
*   **Anomaly Detection Logic**: Analysis showed that 160 outliers represented a stable boundary; further adjustments yielded diminishing returns, confirming this as the optimal threshold.
*   **Class-Specific Imputation**: Strategy aligned with established coding guidelines for robust data handling.
*   **Validation Strategy**: Model reliability was verified using **K-Fold Cross-Validation**, ensuring results are not due to over-fitting on a specific split.

---

## 📚 References & Resources

1. **Akshay**, "Gaussian Naive Bayes with Hyperparameter Tuning," *Analytics Vidhya*, Jan. 27, 2021. [Link](https://www.analyticsvidhya.com/blog/2021/01/gaussian-naive-bayes-with-hyperpameter-tuning/) (accessed Oct. 8, 2024).
2. **K. Zhou**, "Week 2 Coding Guide," *Learn.UQ.*, 2024. [Link](https://learn.uq.edu.au/)
3. **Mnassrib**, "Titanic: logistic regression with python," *Kaggle*, Jan. 07, 2020. [Link](https://www.kaggle.com/code/mnassrib/titanic-logistic-regression-with-python#4.-Logistic-Regression-and-Results) (accessed Oct. 16, 2024).
4. "A comprehensive guide to K-Fold cross validation," *Datacamp*, 2024. [Link](https://www.datacamp.com/tutorial/k-fold-cross-validation) (accessed Oct. 14, 2024).
5. **M. Nashaat**, "Hyperparameter Tuning with GridSearchCV - Mohammed Nashaat - Medium," *Medium*, Oct. 22, 2023. [Link](https://medium.com/@mohammednashaat29/hyperparameter-tuning-with-gridsearchcv-8724f215a383) (accessed Oct. 12, 2024).
6. "Pandas Tutorial: DataFrames in Python," *Datacamp*, 2022. [Link](https://www.datacamp.com/tutorial/pandas-tutorial-dataframe-python) (accessed Oct. 13, 2024).
7. "Understanding cross_val_score | Kaggle." [Link](https://www.kaggle.com/discussions/getting-started/59719) (accessed Oct. 13, 2024).
8. "pandas.DataFrame.iloc — pandas 2.2.3 documentation." [Link](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html) (accessed Oct. 13, 2024).
9. "W3Schools.com." [Link](https://www.w3schools.com/python/pandas/ref_df_iloc.asp) (accessed Oct. 13, 2024).
10. **Ahmedezzatibrahem**, "Titanic Data Cleaning, Visualization, and Modeling," *Kaggle*, Sep. 24, 2024. [Link](https://www.kaggle.com/code/ahmedezzatibrahem/titanic-data-cleaning-visualization-and-modeling) (accessed Oct. 13, 2024).
11. "NumPy: the absolute basics for beginners — NumPy v2.2.dev0 Manual." [Link](https://numpy.org/devdocs/user/absolute_beginners.html)
12. **Rahul**, "Tune Hyperparameters with GridSearchCV," *Analytics Vidhya*, Oct. 11, 2024. [Link](https://www.analyticsvidhya.com/blog/2021/06/tune-hyperparameters-with-gridsearchcv/) (accessed Oct. 13, 2024).
13. **Afnan**, "titanic-first-notebook," *Kaggle*, Oct. 13, 2024. [Link](https://www.kaggle.com/code/afnan970/titanic-first-notebook#Preprocessing) (accessed Oct. 16, 2024).
14. **Noussairmighri**, "House price prediction: Top 4 %," *Kaggle*, Oct. 11, 2024. [Link](https://www.kaggle.com/code/noussairmighri/house-price-prediction-top-4) (accessed Oct. 16, 2024).
15. **Sinakhorami**, "Titanic best working Classifier," *Kaggle*, Dec. 18, 2016. [Link](https://www.kaggle.com/code/sinakhorami/titanic-best-working-classifier) (accessed Oct. 16, 2024).
16. "What are the most effective ways to choose a classification model?," Oct. 24, 2023. [Link](https://www.linkedin.com/advice/0/what-most-effective-ways-choose-classification-model-snmle)
17. **Satishgunjal**, "Tutorial: K Fold Cross Validation," *Kaggle*, Dec. 06, 2020. [Link](https://www.kaggle.com/code/satishgunjal/tutorial-k-fold-cross-validation) (accessed Oct. 13, 2024).
18. **Vinayak123tyagi**, "Novelty detection with Local Outlier Factor," *Kaggle*, Jan. 12, 2020. [Link](https://www.kaggle.com/code/vinayak123tyagi/novelty-detection-with-local-outlier-factor)
19. **J. Korstanje**, "The K-Nearest Neighbors (KNN) algorithm in Python," Sep. 01, 2022. [Link](https://realpython.com/knn-python/)
20. **Prashant**, "Random Forest Classifier tutorial," *Kaggle*, Mar. 13, 2020. [Link](https://www.kaggle.com/code/prashant111/random-forest-classifier-tutorial)
21. **Prashant**, "Decision-Tree Classifier tutorial," *Kaggle*, Mar. 13, 2020. [Link](https://www.kaggle.com/code/prashant111/decision-tree-classifier-tutorial)
