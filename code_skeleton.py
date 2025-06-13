import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures, StandardScaler
from Random_Forest import train_random_forest
from Base_model import base_model
from XGB_model import train_xgboost_model
from svm import train_svm
# from tensorflow_model import tensorflow_model
import xgboost as xgb

def add_sensor_feature(X):
    km = KMeans(n_clusters=3, random_state=42)
    feature_13 = km.fit_predict(X)
    X_sensor = X.copy()
    X_sensor['sensor1'] = (feature_13 == 0).astype(int)
    X_sensor['sensor2'] = (feature_13 == 1).astype(int)
    X_sensor['sensor3'] = (feature_13 == 2).astype(int)
    return X_sensor

def predict(X_test):
    # TODO replace this with your model's predictions
    # For now, we will just return random predictions
    labels = np.random.randint(4, size=len(X_test))
    outliers = np.random.randint(2, size=len(X_test))
    return labels, outliers


def generate_submission(test_data):
    label_predictions, outlier_predictions = predict(test_data)
    
    # IMPORTANT: stick to this format for the submission, 
    # otherwise your submission will results in an error
    submission_df = pd.DataFrame({ 
        "id": test_data["id"],
        "label": label_predictions,
        "outlier": outlier_predictions
    })
    return submission_df

def main():
    df = pd.read_csv("D.csv")
    X = df.iloc[:,1:13]
    y = df.iloc[:, -1]

    # col_to_drop = X.columns[5]
    # X = X.drop(columns=[col_to_drop])  

    # X_sensor = add_sensor_feature(X)
    # plt.scatter(X_sensor["feature_0"], X_sensor["feature_1"], c=X_sensor["sensor"])
    # plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
    ("power", PowerTransformer(method="yeo-johnson")),
    ("poly",  PolynomialFeatures(degree=3, include_bias=True)),
    ("scale", StandardScaler()),
    ("model", xgb.XGBClassifier(
         n_estimators=100,
         max_depth=8,
         learning_rate=0.1,
         objective="multi:softmax",
         num_class=4,
         random_state=42,
        )),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = pipeline.score(X_test, y_test)

    print(f"Pipeline Accuracy: {accuracy}")
    print(f"Recall Score: {recall_score(y_test, y_pred, average=None)}")
    print(f"Precision Score: {precision_score(y_test, y_pred, average=None)}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average=None)}")


    # base_model(X, y)

    # results = tensorflow_model(X, y, test_size=0.2, epochs=200, batch_size=1048, random_state=42)
    # print(f"Tensorflow: {results}")

    # rf_model, metrics = train_random_forest(X_train, y_train, X_test, y_test)

    # rf_model1, metrics1 = train_svm(X_train, y_train, X_test, y_test)

    # train_xgboost_model(X, y)

    df_leaderboard = pd.read_csv("D_test_leaderboard.csv")
    submission_df = generate_submission(df_leaderboard)
    # IMPORTANT: The submission file must be named "submission_leaderboard_GroupName.csv",
    # replace GroupName with a group name of your choice. If you do not provide a group name, 
    # your submission will fail!
    submission_df.to_csv("submission_leaderboard_GroupName.csv", index=False)
    

    # For the final leaderboard, change the file name to "submission_final_GroupName.csv"
    df_final = pd.read_csv("D_test_final.csv")
    submission_df = generate_submission(df_final)
    submission_df.to_csv("submission_final_GroupName.csv", index=False)

if __name__ == "__main__":
    main()
