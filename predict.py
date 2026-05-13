import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score

def test(file: str):
    # Load the test data
    df_test = pd.read_csv(file)

    # Scale the features of the test data
    scaler = MinMaxScaler()
    X_test = df_test.values
    X_test_scaled = scaler.fit_transform(X_test)

    # Load clean training data
    clean_train_data = pd.read_csv('./clean_train_data.csv')
    X_train = clean_train_data.drop("Target (Col 106)", axis=1)
    y_train = clean_train_data["Target (Col 106)"]

    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    # Train the KNN model
    knn = KNeighborsClassifier(n_neighbors=9, weights='uniform', metric='euclidean')
    knn.fit(X_train, y_train)

    # Make predictions with KNN on the cleaned test data
    y_pred_knn = knn.predict(X_test_scaled)

    # Calculate accuracy and F1 score on the train data
    accuracy = cross_val_score(knn, X_train, y_train, cv=kf, scoring='accuracy')
    accuracy_final = "{:.3f}".format(accuracy.mean())
    f1 = cross_val_score(knn, X_train, y_train, cv=kf, scoring='f1')
    f1_final = "{:.3f}".format(f1.mean())

    # Format predictions for output
    y_pred_str = "\n".join(f"{pred}," for pred in y_pred_knn)

    # Write predictions and scores to the output file
    with open('./test_data_prediction.csv', 'w') as f:
        f.write(y_pred_str + "\n" + f"{accuracy_final}, {f1_final}")

if __name__ == "__main__":
    test('./test_data.csv')