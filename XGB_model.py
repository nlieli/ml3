import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score, f1_score

def train_xgboost_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
        test_size=0.3, random_state=42)

    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=15,
        learning_rate=0.15,
        objective='multi:softprob',
        num_class=4,
        min_child_weight=1,
        reg_alpha=0.1,
        reg_lambda=0.1,
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
    return None

