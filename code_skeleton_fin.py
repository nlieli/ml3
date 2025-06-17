import pandas as pd
import numpy as np

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
