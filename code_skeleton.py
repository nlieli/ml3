import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures, StandardScaler
from Random_Forest import train_random_forest
from Base_model import base_model
from XGB_model import train_xgboost_model
from svm import train_svm
from gradient_descent import train_gradient_descent_model
# from tensorflow_model import tensorflow_model
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
import umap
from sklearn.utils.class_weight import compute_class_weight



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
    df = pd.read_csv("D.csv")
    X = df.iloc[:, 1:13]
    y = df.iloc[:, -1]
    X_test = X_test.iloc[:, 1:13]
    gmm, threshold = train_outlier_gmm(X, outlier_fraction=0.2)
    outlier_mask_train = predict_outliers_gmm(gmm, threshold, X)
    X_without_outliers = X[~outlier_mask_train]
    y_without_outliers = y[~outlier_mask_train]


    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_without_outliers), y=y_without_outliers)
    class_weights_dict = dict(enumerate(class_weights))
    sample_weights = np.array([class_weights_dict[label] for label in y_without_outliers])

    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=15,
        objective='multi:softmax',
        num_class=4,
        random_state=42 
        )
    
    #xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
    #f1 = f1_score(y_test1, xgb_model.predict(X_test1), average='macro')
    #print(f"Final F1 = {f1}")
    xgb_model.fit(X_without_outliers, y_without_outliers, sample_weight=sample_weights)

    labels =  xgb_model.predict(X_test)
    outliers = predict_outliers_gmm(gmm, threshold, X_test).astype(int)
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

def train_outlier_gmm(X_train, outlier_fraction=0.20):
    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=0)
    gmm.fit(X_train)
    log_probs = gmm.score_samples(X_train)
    threshold = np.percentile(log_probs, 100 * outlier_fraction)
    return gmm, threshold

def predict_outliers_gmm(gmm, threshold, X_new):
    log_probs_new = gmm.score_samples(X_new)
    return log_probs_new < threshold  # True = Outlier

# def train_outlier_gmm_with_validation(X_train, X_val_out):
#     gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=0)
#     gmm.fit(X_train)
    
#     log_probs_val = gmm.score_samples(X_val_out)
#     threshold = np.percentile(log_probs_val, 99)  # z.B. 95%-Perzentil der bekannten Outlier
#     return gmm, threshold


def main():
    df = pd.read_csv("D.csv")
    dfo = pd.read_csv("D_out.csv")
    Xo = dfo.iloc[:,1:13]
    X = df.iloc[:,1:13]
    y = df.iloc[:, -1]

    # outlier_class = pd.Series([4] * len(Xo), name='label')
    # ywo1 = pd.concat([y, outlier_class], ignore_index=True)
    # Xwo = pd.concat([X, Xo], ignore_index=True)

    # col_to_drop = X.columns[5]
    # X = X.drop(columns=[col_to_drop])  

    X_sensor = add_sensor_feature(X)
    # print(X_sensor)
    # plt.scatter(X_sensor["feature_0"], X_sensor["feature_1"], c=X_sensor["sensor1"])
    # plt.show()
    # 2. Sensoren kalibrieren (Mittelwert-Normalisierung)


    gmm, threshold = train_outlier_gmm(X, outlier_fraction=0.2)
    outliers = predict_outliers_gmm(gmm, threshold, Xo)
    print(f"Number of outliers detected: {np.sum(outliers)}")

    outlier_mask_train = predict_outliers_gmm(gmm, threshold, X)
    X_without_outliers = X[~outlier_mask_train]
    y_without_outliers = y[~outlier_mask_train]

    X_train, X_test, y_train, y_test = train_test_split(X_without_outliers, y_without_outliers, test_size=0.2, random_state=42)



    '''
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
    print(f"F1 Macro Score: {f1_score(y_test, y_pred, average='macro')}")
    '''



    # base_model(X, y)

    # results = tensorflow_model(X, y, test_size=0.2, epochs=200, batch_size=1048, random_state=42)
    # print(f"Tensorflow: {results}")

    # rf_model, metrics = train_random_forest(X_train, y_train, X_test, y_test)

    # rf_model1, metrics1 = train_svm(X_train, y_train, X_test, y_test)

    # xgb_model, metrics2 = train_xgboost_model(X_train, y_train, X_test, y_test)

    # gd_model, metrics_gd = train_gradient_descent_model(X_train.to_numpy(), y_train, X_test.to_numpy(), y_test)


    df_leaderboard = pd.read_csv("D_test_leaderboard.csv")
    submission_df = generate_submission(df_leaderboard)
    # IMPORTANT: The submission file must be named "submission_leaderboard_GroupName.csv",
    # replace GroupName with a group name of your choice. If you do not provide a group name, 
    # your submission will fail!
    submission_df.to_csv("submission_leaderboard_GanzEgal.csv", index=False)
    

    # For the final leaderboard, change the file name to "submission_final_GroupName.csv"
    df_final = pd.read_csv("D_test_final.csv")
    submission_df = generate_submission(df_final)
    submission_df.to_csv("submission_final_GroupName.csv", index=False)

if __name__ == "__main__":
    main()
