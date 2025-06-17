# general packages 
import numpy as np
import sklearn.metrics as metrics

# model specific
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures, StandardScaler
import xgboost as xgb
from xgboost import XGBClassifier 
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

# outlier detection
from sklearn.mixture import GaussianMixture

# Outlier detection
class Outlier_Detector:
    def __init__(self, X_train: np.ndarray, outlier_fraction: float = 0.2) -> None:
        self.gmm = GaussianMixture(n_components=5, covariance_type="full", random_state=42)
        self.gmm.fit(X_train)
        log_probs: np.ndarray = self.gmm.score_samples(X_train)
        self.threshold = np.percentile(log_probs, 100 * outlier_fraction)

    def predict(self, X: np.ndarray):
        # predicts outliers in dataset X
        log_probs_new = self.gmm.score_samples(X)
        return log_probs_new < self.threshold

    def filter(self, X: np.ndarray, y: np.ndarray = np.ndarray([])) -> np.ndarray:
        # removes all outliers from the data set X 
        mask = self.predict(X)
        X_filtered = X[~mask]
        y_fitlered = np.ndarray([]) 
        if y is not np.empty:
            y_filtered = y[~mask]
            return X_filtered, y_filtered

        return X_filtered

'''
The parent class 'Model' provide general functionality for all kinds of models such as print().
The child classes implement an (almost) uniform API for the different models. 
'''


# Parent Class
class Model:
    def __init__(self, name: str) -> None:
        self.name = name

        # model data
        self.y_test: np.ndarray = None
        self.y_pred: np.ndarray = None

    def print_scores(self, y_test: np.ndarray) -> None:
        self.y_test = y_test
        if self.y_pred is None:
            raise ValueError("Prediction data has not been provided")

        # per class accuracy
        classes = np.unique(self.y_test)
        class_acc = {}
        for c in classes:
            idx = (self.y_test == c)
            acc = metrics.accuracy_score(self.y_test[idx], self.y_pred[idx])
            class_acc[c] = acc

        self.accuracy = metrics.accuracy_score(self.y_test, self.y_pred)
        self.class_accuracy = class_acc
        self.precision = metrics.precision_score(self.y_test, self.y_pred, average=None)
        self.recall = metrics.recall_score(self.y_test, self.y_pred, average=None)
        self.f1 = metrics.f1_score(self.y_test, self.y_pred, average=None)
        self.f1_macro = metrics.f1_score(self.y_test, self.y_pred, average="macro")

        print(f"\n--- {self.name} ---")
        print(f"Accuracy = {self.accuracy}")
        print("Class Accuracy = ", list(self.class_accuracy.values()))
        print(f"Recall = {self.recall}")
        print(f"Precision = {self.precision}")
        print(f"F1 = {self.f1}")
        print(f"F1_macro = {self.f1_macro}")

    # functions for specific model override
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        # implement the model training here - usually using fit 
        ...

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        # implement predict method that also assigns the self.y_pred property
        ...

# Child Classes
class Base_Model(Model):
    def __init__(self, name: str = "Base Model"):
        super().__init__(name)
        self.model = KNeighborsClassifier(n_neighbors=10, weights="distance") 

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        self.y_pred = self.model.predict(X_test)
        return self.y_pred

class RF_Model(Model):
    def __init__(self, name: str = "Random Forest", **kwargs):
        super().__init__(name)
        self.model = RandomForestClassifier(**kwargs)


    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        self.y_pred = self.model.predict(X_test)
        return self.y_pred

class SVM_Model(Model):
    def __init__(self, name: str = "SVM Model", **kwargs):
        super().__init__(name)
        self.kwargs = kwargs

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.model = SVC(**self.kwargs)
        self.model.fit(X_train_scaled, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)
        self.y_pred = self.model.predict(X_test_scaled)
        return self.y_pred

class XGB_Model(Model):
    def __init__(self, name: str = "XGB Model", **kwargs):
        super().__init__(name)
        self.model = XGBClassifier(**kwargs)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
        class_weights_dict = dict(enumerate(class_weights))
        sample_weights = np.array([class_weights_dict[label] for label in y_train])

        self.model.fit(X_train, y_train, sample_weight=sample_weights)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        self.y_pred = self.model.predict(X_test)
        return self.y_pred

class PIP_Model(Model):
    def __init__(self, name: str = "Pipeline Model"):
        super().__init__(name)
        self.model = Pipeline([
        ("power", PowerTransformer(method="yeo-johnson")),
        ("poly",  PolynomialFeatures(degree=3, include_bias=False)),
        ("scale", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif)),
        ("model", XGBClassifier(
             n_estimators=190,
             max_depth=110,
             learning_rate=0.1,
             objective="multi:softmax",
             num_class=4,
             random_state=42,
            )),
        ])

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        gs = GridSearchCV(
            self.model, 
            {
            'select__k': ['all'],
            'model__max_depth': [110],
            },
            cv=2,
            scoring='f1_macro'
          )
        gs.fit(X_train, y_train)
        self.model = gs.best_estimator_
        # self.model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        self.y_pred = self.model.predict(X_test)
        return self.y_pred




