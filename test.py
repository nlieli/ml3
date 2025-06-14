import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class OutlierDetector:
    """
    Multi-method outlier detection class for sensor data analysis
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.thresholds = {}
        self.results = {}
        
    def load_data(self, train_path='D.csv', outlier_path='Dout.csv'):
        """Load training data and known outliers"""
        self.D = pd.read_csv("D.csv")
        self.D_out = pd.read_csv("D_out.csv")
        self.X_out = self.D_out.iloc[:,1:13]
        self.X = self.D.iloc[:,1:13]
        self.y = self.D.iloc[:, -1]
        
        print(f"Training data shape: {self.X.shape}")
        print(f"Known outliers shape: {self.X_out.shape}")
        print(f"Class distribution: {np.bincount(self.y)}")
        
    def preprocess_data(self):
        """Standardize features for outlier detection"""
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.X_out_scaled = self.scaler.transform(self.X_out)
        
    def method_1_gaussian_mixture_model(self, n_components_range=[1, 2, 3, 4, 5]):
        """
        Method 1: Gaussian Mixture Model (GMM) approach
        Fits a probabilistic model and uses likelihood as outlier score
        """
        print("\n=== Method 1: Gaussian Mixture Model ===")
        
        best_aic = np.inf
        best_gmm = None
        best_n_components = None
        
        # Model selection using AIC
        for n_components in n_components_range:
            try:
                gmm = GaussianMixture(n_components=n_components, random_state=42)
                gmm.fit(self.X_scaled)
                aic = gmm.aic(self.X_scaled)
                print(f"Components: {n_components}, AIC: {aic:.2f}")
                
                if aic < best_aic:
                    best_aic = aic
                    best_gmm = gmm
                    best_n_components = n_components
            except Exception as e:
                print(f"Error fitting GMM with {n_components} components: {e}")
                continue
        
        if best_gmm is None:
            print("Error: Could not fit any GMM model. Using single component GMM.")
            best_gmm = GaussianMixture(n_components=1, random_state=42)
            best_gmm.fit(self.X_scaled)
            best_n_components = 1
        
        print(f"Best model: {best_n_components} components (AIC: {best_aic:.2f})")
        
        # Calculate log-likelihood scores
        inlier_scores = best_gmm.score_samples(self.X_scaled)
        outlier_scores = best_gmm.score_samples(self.X_out_scaled)
        
        # Determine threshold using known outliers
        # Use percentile approach: threshold where most known outliers are below
        threshold_candidates = np.percentile(inlier_scores, [1, 5, 10, 15, 20])
        
        best_threshold = None
        best_separation = -np.inf
        
        for thresh in threshold_candidates:
            true_positive_rate = np.mean(outlier_scores <= thresh)  # Known outliers correctly identified
            false_positive_rate = np.mean(inlier_scores <= thresh)  # Inliers incorrectly identified
            separation = true_positive_rate - false_positive_rate
            
            print(f"Threshold: {thresh:.3f}, TPR: {true_positive_rate:.3f}, FPR: {false_positive_rate:.3f}, Sep: {separation:.3f}")
            
            if separation > best_separation:
                best_separation = separation
                best_threshold = thresh
        
        self.models['gmm'] = best_gmm
        self.thresholds['gmm'] = best_threshold
        
        # Apply outlier detection
        outlier_mask = inlier_scores <= best_threshold
        n_detected_outliers = np.sum(outlier_mask)
        
        print(f"Selected threshold: {best_threshold:.3f}")
        print(f"Detected outliers in training data: {n_detected_outliers}/{len(self.X)} ({100*n_detected_outliers/len(self.X):.1f}%)")
        
        self.results['gmm'] = {
            'outlier_mask': outlier_mask,
            'inlier_scores': inlier_scores,
            'outlier_scores': outlier_scores,
            'threshold': best_threshold,
            'n_detected': n_detected_outliers
        }
        
        return outlier_mask
    
    def method_2_isolation_forest(self, contamination=0.2):
        """
        Method 2: Isolation Forest
        Tree-based anomaly detection method
        """
        print("\n=== Method 2: Isolation Forest ===")
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
        outlier_predictions = iso_forest.fit_predict(self.X_scaled)
        
        # Get anomaly scores
        anomaly_scores = iso_forest.decision_function(self.X_scaled)
        outlier_scores_iso = iso_forest.decision_function(self.X_out_scaled)
        
        # Convert predictions to boolean mask (True = outlier)
        outlier_mask = outlier_predictions == -1
        n_detected_outliers = np.sum(outlier_mask)
        
        print(f"Contamination parameter: {contamination}")
        print(f"Detected outliers in training data: {n_detected_outliers}/{len(self.X)} ({100*n_detected_outliers/len(self.X):.1f}%)")
        
        # Evaluate on known outliers
        known_outlier_predictions = iso_forest.predict(self.X_out_scaled)
        known_outliers_detected = np.sum(known_outlier_predictions == -1)
        print(f"Known outliers detected: {known_outliers_detected}/{len(self.X_out)} ({100*known_outliers_detected/len(self.X_out):.1f}%)")
        
        self.models['isolation_forest'] = iso_forest
        self.results['isolation_forest'] = {
            'outlier_mask': outlier_mask,
            'anomaly_scores': anomaly_scores,
            'outlier_scores': outlier_scores_iso,
            'n_detected': n_detected_outliers
        }
        
        return outlier_mask
    
    def method_3_local_outlier_factor(self, n_neighbors=20):
        """
        Method 3: Local Outlier Factor (LOF)
        Density-based outlier detection
        """
        print("\n=== Method 3: Local Outlier Factor ===")
        
        # Fit LOF
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.2)
        outlier_predictions = lof.fit_predict(self.X_scaled)
        
        # Get LOF scores
        lof_scores = -lof.negative_outlier_factor_  # Convert to positive scores
        
        # Convert predictions to boolean mask
        outlier_mask = outlier_predictions == -1
        n_detected_outliers = np.sum(outlier_mask)
        
        print(f"Number of neighbors: {n_neighbors}")
        print(f"Detected outliers in training data: {n_detected_outliers}/{len(self.X)} ({100*n_detected_outliers/len(self.X):.1f}%)")
        
        # For known outliers, we need to use kneighbors
        # LOF doesn't have predict method, so we approximate
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(self.X_scaled)
        
        self.models['lof'] = lof
        self.results['lof'] = {
            'outlier_mask': outlier_mask,
            'lof_scores': lof_scores,
            'n_detected': n_detected_outliers
        }
        
        return outlier_mask
    
    def method_4_one_class_svm(self, nu=0.2):
        """
        Method 4: One-Class SVM
        Support vector-based novelty detection
        """
        print("\n=== Method 4: One-Class SVM ===")
        
        # Fit One-Class SVM
        oc_svm = OneClassSVM(nu=nu, kernel='rbf', gamma='scale')
        oc_svm.fit(self.X_scaled)
        
        # Predict outliers
        outlier_predictions = oc_svm.predict(self.X_scaled)
        decision_scores = oc_svm.decision_function(self.X_scaled)
        
        # Known outlier predictions
        known_outlier_predictions = oc_svm.predict(self.X_out_scaled)
        known_outlier_scores = oc_svm.decision_function(self.X_out_scaled)
        
        # Convert predictions to boolean mask
        outlier_mask = outlier_predictions == -1
        n_detected_outliers = np.sum(outlier_mask)
        
        # Evaluate on known outliers
        known_outliers_detected = np.sum(known_outlier_predictions == -1)
        
        print(f"Nu parameter: {nu}")
        print(f"Detected outliers in training data: {n_detected_outliers}/{len(self.X)} ({100*n_detected_outliers/len(self.X):.1f}%)")
        print(f"Known outliers detected: {known_outliers_detected}/{len(self.X_out)} ({100*known_outliers_detected/len(self.X_out):.1f}%)")
        
        self.models['one_class_svm'] = oc_svm
        self.results['one_class_svm'] = {
            'outlier_mask': outlier_mask,
            'decision_scores': decision_scores,
            'outlier_scores': known_outlier_scores,
            'n_detected': n_detected_outliers
        }
        
        return outlier_mask
    
    def visualize_results(self):
        """Create visualizations for outlier detection results"""
        
        # 1. PCA visualization of outliers
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_scaled)
        X_out_pca = pca.transform(self.X_out_scaled)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Outlier Detection Results', fontsize=16)
        
        methods = ['gmm', 'isolation_forest', 'lof', 'one_class_svm']
        method_names = ['Gaussian Mixture Model', 'Isolation Forest', 'Local Outlier Factor', 'One-Class SVM']
        
        for i, (method, name) in enumerate(zip(methods, method_names)):
            ax = axes[i//2, i%2]
            
            if method in self.results:
                outlier_mask = self.results[method]['outlier_mask']
                
                # Plot inliers and detected outliers
                ax.scatter(X_pca[~outlier_mask, 0], X_pca[~outlier_mask, 1], 
                          c='blue', alpha=0.6, s=20, label='Inliers')
                ax.scatter(X_pca[outlier_mask, 0], X_pca[outlier_mask, 1], 
                          c='red', alpha=0.8, s=30, label='Detected Outliers')
                
                # Plot known outliers
                ax.scatter(X_out_pca[:, 0], X_out_pca[:, 1], 
                          c='orange', marker='x', s=50, label='Known Outliers')
                
                ax.set_title(f'{name}\n({self.results[method]["n_detected"]} detected)')
                ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
                ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 2. Score distributions
        if 'gmm' in self.results:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # GMM likelihood scores
            ax = axes[0]
            ax.hist(self.results['gmm']['inlier_scores'], bins=50, alpha=0.7, 
                   label='Training Data', density=True)
            ax.hist(self.results['gmm']['outlier_scores'], bins=20, alpha=0.7, 
                   label='Known Outliers', density=True)
            ax.axvline(self.results['gmm']['threshold'], color='red', linestyle='--', 
                      label=f'Threshold = {self.results["gmm"]["threshold"]:.3f}')
            ax.set_xlabel('Log-Likelihood Score')
            ax.set_ylabel('Density')
            ax.set_title('GMM: Score Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Isolation Forest scores
            if 'isolation_forest' in self.results:
                ax = axes[1]
                ax.hist(self.results['isolation_forest']['anomaly_scores'], bins=50, alpha=0.7, 
                       label='Training Data', density=True)
                ax.hist(self.results['isolation_forest']['outlier_scores'], bins=20, alpha=0.7, 
                       label='Known Outliers', density=True)
                ax.set_xlabel('Anomaly Score')
                ax.set_ylabel('Density')
                ax.set_title('Isolation Forest: Score Distribution')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    def compare_methods(self):
        """Compare different outlier detection methods"""
        print("\n=== Method Comparison ===")
        
        comparison_data = []
        for method_name, result in self.results.items():
            n_detected = result['n_detected']
            percentage = 100 * n_detected / len(self.X)
            
            comparison_data.append({
                'Method': method_name.replace('_', ' ').title(),
                'Outliers Detected': n_detected,
                'Percentage': f"{percentage:.1f}%"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def get_clean_dataset(self, method='gmm'):
        """
        Get cleaned dataset with outliers removed
        
        Args:
            method: Which outlier detection method to use ('gmm', 'isolation_forest', 'lof', 'one_class_svm')
        
        Returns:
            X_clean, y_clean: Features and labels with outliers removed
        """
        if method not in self.results:
            raise ValueError(f"Method '{method}' not found. Available methods: {list(self.results.keys())}")
        
        outlier_mask = self.results[method]['outlier_mask']
        inlier_mask = ~outlier_mask
        
        X_clean = self.X[inlier_mask]
        y_clean = self.y[inlier_mask]
        
        print(f"\nDataset cleaning using {method}:")
        print(f"Original dataset size: {len(self.X)}")
        print(f"Cleaned dataset size: {len(X_clean)}")
        print(f"Removed outliers: {np.sum(outlier_mask)}")
        print(f"Class distribution after cleaning: {np.bincount(y_clean)}")
        
        return X_clean, y_clean
    
    def run_all_methods(self):
        """Run all outlier detection methods"""
        print("Running all outlier detection methods...")
        
        # Preprocess data
        self.preprocess_data()
        
        # Run all methods
        self.method_1_gaussian_mixture_model()
        self.method_2_isolation_forest()
        self.method_3_local_outlier_factor()
        self.method_4_one_class_svm()
        
        # Visualize and compare
        self.visualize_results()
        self.compare_methods()

# Example usage and demonstration
def demonstrate_outlier_detection():
    """
    Demonstration function showing how to use the OutlierDetector class
    """
    print("=== Outlier Detection for Sensor Data ===")
    print("This implementation provides multiple methods for detecting outliers in sensor data.")
    
    # Initialize detector
    detector = OutlierDetector()
    detector.load_data('D.csv', 'D_out.csv')
    detector.run_all_methods()
    
    # Note: You would load your actual data files here
    # detector.load_data('D.csv', 'Dout.csv')
    
    print("\nTo use this implementation:")
    print("1. Load your data: detector.load_data('D.csv', 'Dout.csv')")
    print("2. Run all methods: detector.run_all_methods()")
    print("3. Get clean dataset: X_clean, y_clean = detector.get_clean_dataset('gmm')")
    print("4. Retrain your best model on the cleaned data")
    
    print("\nMethods implemented:")
    print("- Gaussian Mixture Model (GMM): Probabilistic approach using likelihood")
    print("- Isolation Forest: Tree-based ensemble method")
    print("- Local Outlier Factor (LOF): Density-based approach")
    print("- One-Class SVM: Support vector machine for novelty detection")

if __name__ == "__main__":
    demonstrate_outlier_detection()