rf_models_config = [
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


svm_models_config = [
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

xgb_models_config = [
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



