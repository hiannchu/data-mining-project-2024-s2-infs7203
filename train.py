import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import f1_score
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer


np.random.seed(0)

# Load train data
df = pd.read_csv('./DM_Project_24.csv')

# Missing value imputation start from here
print('Missing value imputation...')

# Split the data into features and target
X = df.iloc[:, :-1]  # Features (Num and Nom columns)
y = df['Target (Col 106)']  # Target column

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Define the imputer for mean and median
mean_imputer = SimpleImputer(strategy='mean')
median_imputer = SimpleImputer(strategy='median')

# Function to evaluate performance with different imputers
def evaluate_imputer(imputer):
    # Impute missing values
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Train the model
    model = DecisionTreeClassifier(max_depth=5, random_state=0)
    model.fit(X_train_imputed, y_train)
    
    # Predict and evaluate using F1-score
    predictions = model.predict(X_test_imputed)
    f1 = f1_score(y_test, predictions)
    return f1

# Evaluate using mean imputer
mean_f1 = evaluate_imputer(mean_imputer)
print(f'Mean Imputation F1-score: {mean_f1:.3f}')

# Evaluate using median imputer
median_f1 = evaluate_imputer(median_imputer)
print(f'Median Imputation F1-score: {median_f1:.3f}')

if mean_f1 > median_f1:
    print('Recommended Method: Mean Imputation')
else:
    print('Recommended Method: Median Imputation')

print('================================')

# Use median imputation for numerical columns, and most frequently imputation for nominal columns
# Imputation by class-specific values
df_impu_class = df.copy()
if df.isnull().values.any():
    cat_list = df_impu_class.iloc[:, 105].unique()
    for cat in cat_list:
        df_impu_class.loc[df_impu_class.iloc[:, 105] == cat, df_impu_class.columns[:103]] = \
            df_impu_class.loc[df_impu_class.iloc[:, 105] == cat, df_impu_class.columns[:103]].fillna(
                df_impu_class.loc[df_impu_class.iloc[:, 105] == cat, df_impu_class.columns[:103]].median()
            )
        df_impu_class.loc[df_impu_class.iloc[:, 105] == cat, df_impu_class.columns[103]] = \
            df_impu_class.loc[df_impu_class.iloc[:, 105] == cat, df_impu_class.columns[103]].fillna(
                df_impu_class.loc[df_impu_class.iloc[:, 105] == cat, df_impu_class.columns[103]].mode().iloc[0]
            )
        df_impu_class.loc[df_impu_class.iloc[:, 105] == cat, df_impu_class.columns[104]] = \
            df_impu_class.loc[df_impu_class.iloc[:, 105] == cat, df_impu_class.columns[104]].fillna(
                df_impu_class.loc[df_impu_class.iloc[:, 105] == cat, df_impu_class.columns[104]].mode().iloc[0]
            )
        
print('Missing value imputation done.')
print('================================')

# Anomaly detection start from here
print('Anomaly detection...')

df_anomaly_detection = df_impu_class.copy()
X = df_anomaly_detection.iloc[:, :105].values
y = df_anomaly_detection.iloc[:, -1].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print('See LOF performance...')
# Define different values for n_neighbors
n_neighbors_values_lof = [10, 30, 50, 100, 150, 200, 250, 300, 350, 400]

for n_neighbors_lof in n_neighbors_values_lof:
    # Train the LOF model
    clf = LocalOutlierFactor(n_neighbors=n_neighbors_lof, novelty=True, contamination=0.1)
    clf.fit(X_scaled)

    # Make predictions
    y_pred = clf.predict(X_scaled)

    # Calculate the number of detected outliers
    n_error_outliers_lof = np.sum(y_pred == -1)  # Count of detected outliers

    # Output the number of detected outliers
    print(f"n_neighbors={n_neighbors_lof}, Detected outliers: {n_error_outliers_lof}")

print('See Isolation Forest performance...')
# Define different values for n_estimators and max_samples
n_estimators_if = [10, 30, 50, 100, 150, 200]
max_samples_if = [5, 10, 50, 100, 150]

# Loop through different n_estimators and max_samples
for estimators in n_estimators_if:
    for max_samples in max_samples_if:
        # Train the Isolation Forest model
        clf_if = IsolationForest(n_estimators=estimators, max_samples=max_samples, random_state=0, contamination=0.1)
        clf_if.fit(X_scaled)

        # Make predictions
        y_pred_if = clf_if.predict(X_scaled)

        # Calculate the number of detected outliers
        n_error_outliers_if = np.sum(y_pred_if == -1)  # Count of detected outliers

        # Output the number of detected outliers
        print(f"n_estimators={estimators}, max_samples={max_samples}, Detected outliers: {n_error_outliers_if}")

print('Through the output above, the maximum number of detected outliers is 160.')
print('Choose Isolation Forest because it offers more flexibility in setting parameters. In Isolation Forest, select n_estimators=10 and max_samples=50 to detect 160 outliers.')

# Train the Isolation Forest model
clf_if = IsolationForest(n_estimators=10, max_samples=50, random_state=0, contamination=0.1)
y_pred_final = clf_if.fit_predict(X_scaled)

# Calculate the number of detected outliers
outliers_index = np.where(y_pred_final == -1)[0]

# Remove the outliers
df_cleaned = df_anomaly_detection.drop(index=outliers_index)

print('Anomaly detection done.')

print('================================')

# Normalization start here
# Code from week 2 coding guide
# Create a copy of the cleaned DataFrame for normalization
df_class_minmax = df_cleaned.copy()

# Initialize the MinMaxScaler to scale features to a range [0, 1]
scaler = MinMaxScaler()

# Fit the scaler on the numerical columns (first 103 columns)
scaler.fit(df_class_minmax.loc[:, df_cleaned.columns[:103]])

# Transform the numerical columns using the fitted scaler
df_class_minmax.loc[:, df_cleaned.columns[:103]] = \
    scaler.transform(df_class_minmax.loc[:, df_cleaned.columns[:103]])

print('Normalization done.')
print('================================')

print('Export the clean train data to csv...')
df_class_minmax.to_csv('./clean_train_data.csv', index=False)
print('Export done, please check clean_train_data.csv')
print('================================')

# Classifion selection start from here
print('Classifion selection...')

df_new = df_class_minmax.copy()
X = df_new.drop("Target (Col 106)", axis=1)  # Features
y = df_new['Target (Col 106)']  # Target variable

# Initialize KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# Dictionaries to store results and best parameters
results = {}
best_parameters = {}

# Decision Tree
print('Decision Tree...')
param_grid_dt = {
    'max_depth': [1, 10, 20],
    'min_samples_split': [2, 5, 10],
}
dt = DecisionTreeClassifier(random_state=0)
grid_search_dt = GridSearchCV(estimator=dt, param_grid=param_grid_dt, scoring='f1', cv=kf)
grid_search_dt.fit(X, y)  # Fit the model with GridSearchCV

best_model_dt = grid_search_dt.best_estimator_  # Best decision tree model
f1_scores_dt = cross_val_score(best_model_dt, X, y, cv=kf, scoring='f1')  # F1 scores
acc_scores_dt = cross_val_score(best_model_dt, X, y, cv=kf, scoring='accuracy')  # Accuracy scores

# Store results
results['Decision Tree'] = {
    'accuracy': np.mean(acc_scores_dt),  # Mean accuracy
    'f1_score': np.mean(f1_scores_dt)  # Mean F1 score
}
best_parameters['Decision Tree'] = grid_search_dt.best_params_  # Best parameters for decision tree

# Print Decision Tree results
print("Decision Tree - Best Accuracy: {:.3f}, Best F1 Score: {:.3f}, Best Parameters: {}".format(
    results['Decision Tree']['accuracy'], 
    results['Decision Tree']['f1_score'], 
    best_parameters['Decision Tree']
))

# Random Forest
print('Random Forest...')
param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
rf = RandomForestClassifier(random_state=0)
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, scoring='f1')
grid_search_rf.fit(X, y)  # Fit the model with GridSearchCV

best_model_rf = grid_search_rf.best_estimator_  # Best random forest model
f1_scores_rf = cross_val_score(best_model_rf, X, y, cv=kf, scoring='f1')  # F1 scores
acc_scores_rf = cross_val_score(best_model_rf, X, y, cv=kf, scoring='accuracy')  # Accuracy scores

# Store results
results['Random Forest'] = {
    'accuracy': np.mean(acc_scores_rf),  # Mean accuracy
    'f1_score': np.mean(f1_scores_rf)  # Mean F1 score
}
best_parameters['Random Forest'] = grid_search_rf.best_params_  # Best parameters for random forest

# Print Random Forest results
print("Random Forest - Best Accuracy: {:.3f}, Best F1 Score: {:.3f}, Best Parameters: {}".format(
    results['Random Forest']['accuracy'], 
    results['Random Forest']['f1_score'], 
    best_parameters['Random Forest']
))

# KNN
print('KNN...')
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
knn = KNeighborsClassifier()
grid_search_knn = GridSearchCV(estimator=knn, param_grid=param_grid_knn, cv=5, scoring='f1')
grid_search_knn.fit(X, y)  # Fit the model with GridSearchCV

best_model_knn = grid_search_knn.best_estimator_  # Best KNN model
f1_scores_knn = cross_val_score(best_model_knn, X, y, cv=kf, scoring='f1')  # F1 scores
acc_scores_knn = cross_val_score(best_model_knn, X, y, cv=kf, scoring='accuracy')  # Accuracy scores

# Store results
results['KNN'] = {
    'accuracy': np.mean(acc_scores_knn),  # Mean accuracy
    'f1_score': np.mean(f1_scores_knn)  # Mean F1 score
}
best_parameters['KNN'] = grid_search_knn.best_params_  # Best parameters for KNN

# Print KNN results
print("KNN - Best Accuracy: {:.3f}, Best F1 Score: {:.3f}, Best Parameters: {}".format(
    results['KNN']['accuracy'], 
    results['KNN']['f1_score'], 
    best_parameters['KNN']
))

# Naive Bayes
print('Naive Bayes...')
gnb = GaussianNB()
f1_scores_gnb = cross_val_score(gnb, X, y, cv=kf, scoring='f1')  # F1 scores
acc_scores_gnb = cross_val_score(gnb, X, y, cv=kf, scoring='accuracy')  # Accuracy scores

# Store results
results['Naive Bayes'] = {
    'accuracy': np.mean(acc_scores_gnb),  # Mean accuracy
    'f1_score': np.mean(f1_scores_gnb)  # Mean F1 score
}
best_parameters['Naive Bayes'] = None  # Naive Bayes does not have tunable parameters

# Print Naive Bayes results
print("Naive Bayes - Best Accuracy: {:.3f}, Best F1 Score: {:.3f}, Best Parameters: None".format(
    results['Naive Bayes']['accuracy'], 
    results['Naive Bayes']['f1_score']
))

# Determine the best classifier based on F1 score and accuracy
best_classifier = max(results, key=lambda k: (results[k]['f1_score'], results[k]['accuracy']))

# Print the best classifier and its overall results
print("Best Classifier:", best_classifier)
print("Best Parameters:", best_parameters[best_classifier])
print("Accuracy: {:.3f}".format(results[best_classifier]['accuracy']))
print("F1 Score: {:.3f}".format(results[best_classifier]['f1_score']))

# Prepare data for return
X_train = df_new.iloc[:, :103].values  # Features for training
y_train = df_new.iloc[:, -1].values  # Target variable for training
accuracy = "{:.3f}".format(results[best_classifier]['accuracy'])  # Best accuracy
f1_score = "{:.3f}".format(results[best_classifier]['f1_score'])  # Best F1 score



