
def train_svm(X_train, y_train, X_test, y_test):
    """
    Trainiert und vergleicht drei SVM Modelle mit verschiedenen Hyperparametern
    """
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    # Features standardisieren (wichtig für SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Drei verschiedene Hyperparameter-Konfigurationen
    models_config = [
        {
            'name': 'SVM_Linear',
            'params': {
                'kernel': 'linear',
                'C': 10,
                'class_weight': 'balanced',
                'random_state': 42
            }
        },
        {
            'name': 'SVM_RBF_Balanced',
            'params': {
                'kernel': 'rbf',
                'C': 10,
                'gamma': 'scale',
                'class_weight': 'balanced',
                'random_state': 42
            }
        },
        {
            'name': 'SVM_RBF_Complex',
            'params': {
                'kernel': 'rbf',
                'C': 10.0,
                'gamma': 'auto',
                'class_weight': 'balanced',
                'random_state': 42
            }
        }
    ]
    
    results = {}
    trained_models = {}
    
    print("=" * 60)
    print("SUPPORT VECTOR MACHINE MODEL COMPARISON")
    print("=" * 60)
    
    for config in models_config:
        model_name = config['name']
        params = config['params']
        
        print(f"\n--- Training {model_name} ---")
        print(f"Parameters: {params}")
        
        # Modell erstellen und trainieren
        svm_model = SVC(**params)
        svm_model.fit(X_train_scaled, y_train)
        
        # Vorhersagen machen
        y_pred = svm_model.predict(X_test_scaled)
        
        # Metriken berechnen
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=None, zero_division=0)
        recall = recall_score(y_test, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'f1_macro': f1_macro,
            'parameters': params
        }
        
        # Ergebnisse speichern
        results[model_name] = metrics
        trained_models[model_name] = {'model': svm_model, 'scaler': scaler}
        
        # Ergebnisse ausgeben
        print(f"Accuracy = {accuracy:.4f}")
        print(f"F1_macro = {f1_macro:.4f}")
        print(f"Precision = {precision}")
        print(f"Recall = {recall}")
        print(f"F1 = {f1}")

    return {
        'models': trained_models,
        'results': results
    }