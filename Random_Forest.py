
def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Trainiert und vergleicht drei Random Forest Modelle mit verschiedenen Hyperparametern
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Drei verschiedene Hyperparameter-Konfigurationen
    models_config = [
        {
            'name': 'RF_Conservative',
            'params': {
                'n_estimators': 50,
                'max_depth': 5,
                'min_samples_split': 5,
                'min_samples_leaf': 5,
                'random_state': 42
            }
        },
        {
            'name': 'RF_Balanced',
            'params': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 3,
                'min_samples_leaf': 3,
                'random_state': 42
            }
        },
        {
            'name': 'RF_Complex',
            'params': {
                'n_estimators': 200,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
        }
    ]
    
    results = {}
    trained_models = {}
    
    print("=" * 60)
    print("RANDOM FOREST MODEL COMPARISON")
    print("=" * 60)
    
    for config in models_config:
        model_name = config['name']
        params = config['params']
        
        print(f"\n--- Training {model_name} ---")
        print(f"Parameters: {params}")
        
        # Modell erstellen und trainieren
        rf_model = RandomForestClassifier(**params)
        rf_model.fit(X_train, y_train)
        
        # Vorhersagen machen
        y_pred = rf_model.predict(X_test)
        
        # Metriken berechnen
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
            'parameters': params
        }
        
        # Ergebnisse speichern
        results[model_name] = metrics
        trained_models[model_name] = rf_model
        
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
