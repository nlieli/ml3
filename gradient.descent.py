# filename: gradient_descent_model.py

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def softmax(z):
    z -= np.max(z, axis=1, keepdims=True)  # für numerische Stabilität
    exp_scores = np.exp(z)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def one_hot_encode(y, num_classes):
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

def train_gradient_descent_model(X_train, y_train, X_test, y_test, num_classes=4, lr=0.1, epochs=1000, reg=0.01):
    """
    Manuelles Training einer multinomialen logistischen Regression mit Gradient Descent
    """
    np.random.seed(42)
    X_train = np.c_[np.ones(X_train.shape[0]), X_train]  # Bias-Term
    X_test = np.c_[np.ones(X_test.shape[0]), X_test]
    y_one_hot = one_hot_encode(y_train.to_numpy(), num_classes)
    
    n_features = X_train.shape[1]
    W = np.random.randn(n_features, num_classes) * 0.01

    for epoch in range(epochs):
        logits = X_train @ W
        probs = softmax(logits)
        loss = -np.mean(np.sum(y_one_hot * np.log(probs + 1e-15), axis=1)) + reg * np.sum(W * W)
        
        grad_W = X_train.T @ (probs - y_one_hot) / X_train.shape[0] + 2 * reg * W
        W -= lr * grad_W

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # Prediction & Evaluation
    logits_test = X_test @ W
    y_pred = np.argmax(softmax(logits_test), axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_macro": f1_macro
    }

    print("\nGradient Descent Model Evaluation")
    print("---------------------------------")
    print(f"Accuracy: {accuracy}")
    print(f"Recall Score: {recall}")
    print(f"Precision Score: {precision}")
    print(f"F1 Score: {f1}")
    print(f"F1 Macro Score: {f1_macro}")

    return W, metrics