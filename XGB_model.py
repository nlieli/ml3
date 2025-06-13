import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def train_xgboost_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))
    sample_weights = np.array([class_weights_dict[label] for label in y_train])

    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=15,
        learning_rate=0.1,
        objective="multi:softmax",
        num_class=4,
        random_state=42,
    )

    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    accuracy = xgb_model.score(X_test, y_test)
    precision_macro = precision_score(y_test, y_pred, average=None)
    recall_macro = recall_score(y_test, y_pred, average=None)
    f1_macro = f1_score(y_test, y_pred, average=None)
    print(f"\n-------XGBoost Model Results-------")
    print(f"XGBoost Model Accuracy: {accuracy}")
    print(f"XGBoost precision_macro: {precision_macro}")
    print(f"XGBoost recall_macro: {recall_macro}")
    print(f"XGBoost f1_macro: {f1_macro}")

