import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# custom classes
from models import Model, Outlier_Detector, Base_Model, RF_Model, SVM_Model, XGB_Model, PIP_Model

'''
model configs:
Different hyperparameters for the models can be selected by changing the rf_hyper, svm_hyper, ect.
variable. For some models, the hyperparameters need to be selected in the definition of the models
itself.
'''
from model_configs import rf_models_config, svm_models_config, xgb_models_config

def predict(X_test, model: Model, outlier_model: Outlier_Detector):
    # TODO replace this with your model's predictions
    # For now, we will just return random predictions
    X_test = X_test.iloc[:, 1:13]  

    labels = model.predict(X_test)
    outliers = outlier_model.predict(X_test).astype(int)
    return labels, outliers


def generate_submission(test_data, model: Model, outlier_model: Outlier_Detector):
    label_predictions, outlier_predictions = predict(test_data, model, outlier_model)
    
    # IMPORTANT: stick to this format for the submission, 
    # otherwise your submission will results in an error
    submission_df = pd.DataFrame({ 
        "id": test_data["id"],
        "label": label_predictions,
        "outlier": outlier_predictions
    })
    return submission_df


def main():
    # load data
    df = pd.read_csv("D.csv")
    dfo = pd.read_csv("D_out.csv")
    X_outlier = dfo.iloc[:,1:13]
    X = df.iloc[:,1:13]
    y = df.iloc[:, -1]

    # outlier detection
    od = Outlier_Detector(X, 0.2)
    X_without_outliers, y_without_outliers = od.filter(X, y)

    # validation strategy 
    X_train, X_test, y_train, y_test = train_test_split(X_without_outliers, y_without_outliers, test_size=0.3, random_state=42)

    # select model
    model = input("""Select Model: 
        \n [0] Base Model
        \n [1] Random Forest
        \n [2] SVM Model 
        \n [3] XGB Model
        \n [4] Pipeline Model (XGB)

        Input: """)

    model = int(model)
    selected_model: Model = None

    match model:
        case 0:
            # base model
            base_model = Base_Model()
            base_model.train(X_train, y_train)
            base_model.predict(X_test)
            base_model.print_scores(y_test)
            selected_model = base_model

        case 1:
            # random forest model
            rf_hyper: int = 2 # values from 0 to 3 are allowed
            rf_model = RF_Model(name=rf_models_config[rf_hyper]["name"], 
                           **rf_models_config[rf_hyper]["params"])
            rf_model.train(X_train, y_train)
            rf_model.predict(X_test)
            rf_model.print_scores(y_test)
            selected_model = rf_model

        case 2:
            # state vector machine model
            svm_hyper: int = 2
            svm_model = SVM_Model(name=svm_models_config[svm_hyper]["name"],
                              **svm_models_config[svm_hyper]["params"])
            svm_model.train(X_train, y_train)
            svm_model.predict(X_test)
            svm_model.print_scores(y_test)
            selected_model = svm_model

        case 3:
            # xgb model
            xgb_hyper: int = 2
            xgb_model = XGB_Model(name=xgb_models_config[xgb_hyper]["name"],
                                  **xgb_models_config[xgb_hyper]["params"])
            xgb_model.train(X_train, y_train)
            xgb_model.predict(X_test)
            xgb_model.print_scores(y_test)
            selected_model = xgb_model

        case 4:
            # pipeline model
            pip_model = PIP_Model()
            pip_model.train(X_train, y_train)
            pip_model.predict(X_test)
            pip_model.print_scores(y_test)
            selected_model = pip_model

    # Train model on all the data
    selected_model.train(X_without_outliers, y_without_outliers)

    df_leaderboard = pd.read_csv("D_test_leaderboard.csv")
    submission_df = generate_submission(df_leaderboard, selected_model, od)
    # IMPORTANT: The submission file must be named "submission_leaderboard_GroupName.csv",
    # replace GroupName with a group name of your choice. If you do not provide a group name, 
    # your submission will fail!
    submission_df.to_csv("submission_leaderboard_GanzEgal.csv", index=False)
    

    # For the final leaderboard, change the file name to "submission_final_GroupName.csv"
    df_final = pd.read_csv("D_test_final.csv")
    submission_df = generate_submission(df_final, selected_model, od)
    submission_df.to_csv("submission_final_GanzEgal.csv", index=False)


if __name__ == "__main__":
    main()
