import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def train_xgboost_model(X_train, y_train, X_test, y_test):
    """
    Trains and compares multiple XGBoost models with different hyperparameters.
    """

    # Compute class weights
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))
    sample_weights = np.array([class_weights_dict[label] for label in y_train])

    # Define different hyperparameter configurations
    models_config = [
        {
            'name': 'XGB_Conservative',
            'params': {
                'n_estimators': 50,
                'max_depth': 5,
                'learning_rate': 0.05,
                'objective': 'multi:softmax',
                'num_class': 4,
                'random_state': 42
            }
        },
        {
            'name': 'XGB_Balanced',
            'params': {
                'n_estimators': 100,
                'max_depth': 10,
                'learning_rate': 0.1,
                'objective': 'multi:softmax',
                'num_class': 4,
                'random_state': 42
            }
        },
        {
            'name': 'XGB_Complex',
            'params': {
                'n_estimators': 200,
                'max_depth': 15,
                'learning_rate': 0.2,
                'objective': 'multi:softmax',
                'num_class': 4,
                'random_state': 42
            }
        }
    ]

    results = {}
    trained_models = {}

    print("=" * 60)
    print("XGBOOST MODEL COMPARISON")
    print("=" * 60)

    for config in models_config:
        model_name = config['name']
        params = config['params']

        print(f"\n--- Training {model_name} ---")
        print(f"Parameters: {params}")

        # Initialize and train model
        xgb_model = xgb.XGBClassifier(**params)
        xgb_model.fit(X_train, y_train, sample_weight=sample_weights)

        # Predict
        y_pred = xgb_model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=None)
        recall = recall_score(y_test, y_pred, average=None)
        f1 = f1_score(y_test, y_pred, average=None)
        f1_macro = f1_score(y_test, y_pred, average="macro")

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'f1_macro': f1_macro,
            'parameters': params,
            'y_pred': y_pred
        }

        results[model_name] = metrics
        trained_models[model_name] = xgb_model

        # Print results
        print(f"Accuracy = {accuracy:.4f}")
        print(f"F1_macro = {f1_macro:.4f}")
        print(f"Precision = {precision}")
        print(f"Recall = {recall}")
        print(f"F1 = {f1}")

    return {
        'models': trained_models,
        'results': results
    }

