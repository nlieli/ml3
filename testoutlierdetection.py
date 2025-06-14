import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

class OutlierDetector:
    """
    Comprehensive outlier detection class with multiple methods
    """
    
    def __init__(self, method='gmm', contamination=0.1):
        """
        Initialize outlier detector
        
        Args:
            method: 'gmm', 'isolation_forest', 'lof', or 'ensemble'
            contamination: Expected proportion of outliers
        """
        self.method = method
        self.contamination = contamination
        self.detector = None
        self.scaler = StandardScaler()
        self.threshold = None
        
    def fit(self, X, X_outliers=None):
        """
        Fit the outlier detection model
        
        Args:
            X: Training data (inliers)
            X_outliers: Known outliers for threshold determination
        """
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        if self.method == 'gmm':
            self._fit_gmm(X_scaled, X_outliers)
        elif self.method == 'isolation_forest':
            self._fit_isolation_forest(X_scaled)
        elif self.method == 'lof':
            self._fit_lof(X_scaled)
        elif self.method == 'ensemble':
            self._fit_ensemble(X_scaled, X_outliers)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _fit_gmm(self, X_scaled, X_outliers=None):
        """Fit Gaussian Mixture Model"""
        # Determine optimal number of components using BIC
        n_components_range = range(1, min(11, len(X_scaled) // 10))
        bic_scores = []
        
        for n_components in n_components_range:
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(X_scaled)
            bic_scores.append(gmm.bic(X_scaled))
        
        # Select optimal number of components
        optimal_components = n_components_range[np.argmin(bic_scores)]
        
        # Fit final GMM
        self.detector = GaussianMixture(
            n_components=optimal_components, 
            random_state=42
        )
        self.detector.fit(X_scaled)
        
        # Determine threshold using outlier data if available
        if X_outliers is not None:
            self._determine_threshold_with_outliers(X_outliers)
        else:
            # Use percentile-based threshold
            scores = self.detector.score_samples(X_scaled)
            self.threshold = np.percentile(scores, self.contamination * 100)
    
    def _fit_isolation_forest(self, X_scaled):
        """Fit Isolation Forest"""
        self.detector = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        self.detector.fit(X_scaled)
    
    def _fit_lof(self, X_scaled):
        """Fit Local Outlier Factor"""
        self.detector = LocalOutlierFactor(
            contamination=self.contamination,
            novelty=True
        )
        self.detector.fit(X_scaled)
    
    def _fit_ensemble(self, X_scaled, X_outliers=None):
        """Fit ensemble of multiple methods"""
        # Fit multiple detectors
        self.detectors = {
            'gmm': GaussianMixture(n_components=3, random_state=42),
            'isolation_forest': IsolationForest(contamination=self.contamination, random_state=42),
            'lof': LocalOutlierFactor(contamination=self.contamination, novelty=True)
        }
        
        for name, detector in self.detectors.items():
            detector.fit(X_scaled)
        
        # For GMM, determine threshold
        if X_outliers is not None:
            scores = self.detectors['gmm'].score_samples(X_scaled)
            outlier_scores = self.detectors['gmm'].score_samples(
                self.scaler.transform(X_outliers)
            )
            self.threshold = np.percentile(outlier_scores, 95)
        else:
            scores = self.detectors['gmm'].score_samples(X_scaled)
            self.threshold = np.percentile(scores, self.contamination * 100)
    
    def _determine_threshold_with_outliers(self, X_outliers):
        """Determine threshold using known outliers"""
        X_outliers_scaled = self.scaler.transform(X_outliers)
        outlier_scores = self.detector.score_samples(X_outliers_scaled)
        
        # Set threshold at 95th percentile of outlier scores
        self.threshold = np.percentile(outlier_scores, 95)
    
    def predict(self, X):
        """
        Predict outliers in new data
        
        Returns:
            Array of predictions: 1 for inlier, -1 for outlier
        """
        X_scaled = self.scaler.transform(X)
        
        if self.method == 'gmm':
            scores = self.detector.score_samples(X_scaled)
            return np.where(scores > self.threshold, 1, -1)
        
        elif self.method == 'isolation_forest':
            return self.detector.predict(X_scaled)
        
        elif self.method == 'lof':
            return self.detector.predict(X_scaled)
        
        elif self.method == 'ensemble':
            # Majority voting
            predictions = []
            
            # GMM prediction
            gmm_scores = self.detectors['gmm'].score_samples(X_scaled)
            gmm_pred = np.where(gmm_scores > self.threshold, 1, -1)
            predictions.append(gmm_pred)
            
            # Isolation Forest prediction
            iso_pred = self.detectors['isolation_forest'].predict(X_scaled)
            predictions.append(iso_pred)
            
            # LOF prediction
            lof_pred = self.detectors['lof'].predict(X_scaled)
            predictions.append(lof_pred)
            
            # Majority vote
            predictions = np.array(predictions)
            return np.where(np.sum(predictions, axis=0) >= 0, 1, -1)
    
    def get_scores(self, X):
        """Get anomaly scores for data points"""
        X_scaled = self.scaler.transform(X)
        
        if self.method == 'gmm':
            return self.detector.score_samples(X_scaled)
        elif self.method == 'isolation_forest':
            return self.detector.decision_function(X_scaled)
        elif self.method == 'lof':
            return -self.detector.negative_outlier_factor_
        else:
            # For ensemble, return GMM scores
            return self.detectors['gmm'].score_samples(X_scaled)


def evaluate_outlier_detection(detector, X_test, y_true_outliers):
    """
    Evaluate outlier detection performance
    
    Args:
        detector: Fitted OutlierDetector
        X_test: Test features
        y_true_outliers: True outlier labels (1 for inlier, -1 for outlier)
    """
    y_pred = detector.predict(X_test)
    
    # Convert to binary format for metrics
    y_true_binary = (y_true_outliers == 1).astype(int)
    y_pred_binary = (y_pred == 1).astype(int)
    
    print("Outlier Detection Performance:")
    print(classification_report(y_true_binary, y_pred_binary, 
                              target_names=['Outlier', 'Inlier']))
    
    # Confusion matrix
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Outlier', 'Inlier'],
                yticklabels=['Outlier', 'Inlier'])
    plt.title('Outlier Detection Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return y_pred


def compare_model_performance(X_train, y_train, X_test, y_test, 
                            X_train_clean, y_train_clean, model_name="Model"):
    """
    Compare model performance before and after outlier removal
    """
    # Train models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    results = {}
    
    for name, model in models.items():
        # Original model (with outliers)
        model_orig = model.__class__(**model.get_params())
        model_orig.fit(X_train, y_train)
        y_pred_orig = model_orig.predict(X_test)
        
        # Clean model (without outliers)
        model_clean = model.__class__(**model.get_params())
        model_clean.fit(X_train_clean, y_train_clean)
        y_pred_clean = model_clean.predict(X_test)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        results[name] = {
            'original': {
                'accuracy': accuracy_score(y_test, y_pred_orig),
                'precision': precision_score(y_test, y_pred_orig, average='weighted'),
                'recall': recall_score(y_test, y_pred_orig, average='weighted'),
                'f1': f1_score(y_test, y_pred_orig, average='weighted')
            },
            'clean': {
                'accuracy': accuracy_score(y_test, y_pred_clean),
                'precision': precision_score(y_test, y_pred_clean, average='weighted'),
                'recall': recall_score(y_test, y_pred_clean, average='weighted'),
                'f1': f1_score(y_test, y_pred_clean, average='weighted')
            }
        }
    
    # Display results
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print("-" * 40)
        print(f"{'Metric':<12} {'Original':<12} {'Clean':<12} {'Improvement':<12}")
        print("-" * 48)
        
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            orig_val = metrics['original'][metric]
            clean_val = metrics['clean'][metric]
            improvement = clean_val - orig_val
            
            print(f"{metric:<12} {orig_val:<12.4f} {clean_val:<12.4f} {improvement:<12.4f}")
    
    return results


def visualize_outliers(X, outlier_labels, method_name="Outlier Detection"):
    """
    Visualize detected outliers in 2D space (using PCA if needed)
    """
    from sklearn.decomposition import PCA
    
    # Reduce to 2D for visualization if needed
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_viz = pca.fit_transform(X)
        xlabel, ylabel = f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', \
                        f'PC2 ({pca.explained_variance_ratio_[1]:.2%})'
    else:
        X_viz = X
        xlabel, ylabel = 'Feature 1', 'Feature 2'
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    # Plot inliers and outliers
    inliers = outlier_labels == 1
    outliers = outlier_labels == -1
    
    plt.scatter(X_viz[inliers, 0], X_viz[inliers, 1], 
               c='blue', alpha=0.6, s=50, label='Inliers')
    plt.scatter(X_viz[outliers, 0], X_viz[outliers, 1], 
               c='red', alpha=0.8, s=50, label='Outliers', marker='x')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{method_name} Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# Example usage and demonstration
def demonstrate_outlier_detection():
    """
    Demonstrate the outlier detection pipeline with synthetic data
    """
    print("OUTLIER DETECTION DEMONSTRATION")
    print("="*50)
    
    # Generate synthetic dataset
    np.random.seed(42)
    
    # Generate inlier data (normal distribution)
    n_inliers = 800
    X_inliers = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n_inliers)
    y_inliers = (X_inliers[:, 0] + X_inliers[:, 1] + np.random.normal(0, 0.1, n_inliers) > 0).astype(int)
    
    # Generate outlier data
    n_outliers = 100
    X_outliers = np.random.uniform(-4, 4, (n_outliers, 2))
    y_outliers = np.random.randint(0, 2, n_outliers)
    
    # Combine data
    X_all = np.vstack([X_inliers, X_outliers])
    y_all = np.hstack([y_inliers, y_outliers])
    
    # True outlier labels (1 for inlier, -1 for outlier)
    true_outlier_labels = np.hstack([np.ones(n_inliers), -np.ones(n_outliers)])
    
    # Split data
    X_train, X_test, y_train, y_test, true_labels_train, true_labels_test = \
        train_test_split(X_all, y_all, true_outlier_labels, test_size=0.3, random_state=42)
    
    # Test different outlier detection methods
    methods = ['gmm', 'isolation_forest', 'lof', 'ensemble']
    
    for method in methods:
        print(f"\n{'='*20} {method.upper()} {'='*20}")
        
        # Create and fit detector
        detector = OutlierDetector(method=method, contamination=0.1)
        
        # For demonstration, we'll use some known outliers to set threshold
        known_outliers_idx = true_labels_train == -1
        if np.any(known_outliers_idx):
            X_known_outliers = X_train[known_outliers_idx]
            detector.fit(X_train[true_labels_train == 1], X_known_outliers)
        else:
            detector.fit(X_train)
        
        # Predict outliers
        outlier_predictions = detector.predict(X_train)
        
        # Evaluate detection performance
        evaluate_outlier_detection(detector, X_train, true_labels_train)
        
        # Visualize results
        visualize_outliers(X_train, outlier_predictions, f"{method.upper()}")
        
        # Filter dataset to create clean training set
        inlier_mask = outlier_predictions == 1
        X_train_clean = X_train[inlier_mask]
        y_train_clean = y_train[inlier_mask]
        
        print(f"\nDataset size: {len(X_train)} -> {len(X_train_clean)} "
              f"(removed {len(X_train) - len(X_train_clean)} outliers)")
        
        # Compare model performance
        results = compare_model_performance(
            X_train, y_train, X_test, y_test,
            X_train_clean, y_train_clean, method
        )


if __name__ == "__main__":
    # Run demonstration
    demonstrate_outlier_detection()
    
    print("\n" + "="*80)
    print("IMPLEMENTATION COMPLETE")
    print("="*80)
    print("\nTo use with your own data:")
    print("1. Load your dataset D and outlier dataset D_out")
    print("2. Create OutlierDetector with your preferred method")
    print("3. Fit the detector: detector.fit(D, D_out)")
    print("4. Predict outliers: outlier_labels = detector.predict(D)")
    print("5. Filter dataset: D_clean = D[outlier_labels == 1]")
    print("6. Retrain your best model on D_clean")
    print("7. Compare performance using compare_model_performance()")