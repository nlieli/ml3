import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from collections import Counter
from sklearn.mixture import GaussianMixture

class KNNEnsemble:
    """
    Ensemble aus 3 KNN-Modellen mit Mehrheitsvoting
    Wenn mindestens 2 Modelle übereinstimmen -> dieser Wert
    Sonst -> Default-Wert 1
    """
    
    def __init__(self, k_values=[5, 7, 9], default_prediction=1):
        """
        Parameters:
        -----------
        k_values : list
            Liste mit 3 verschiedenen k-Werten für die KNN-Modelle
        default_prediction : int
            Default-Wert wenn keine Mehrheit erreicht wird
        """
        self.k_values = k_values
        self.default_prediction = default_prediction
        self.models = []
        self.scaler = StandardScaler()
        
    def fit(self, X_train, y_train):
        """Trainiert die 3 KNN-Modelle"""
        print("Training KNN Ensemble...")
        print(f"K-Werte: {self.k_values}")
        print(f"Default-Vorhersage: {self.default_prediction}")
        
        # Daten standardisieren
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # 3 KNN-Modelle mit verschiedenen k-Werten trainieren
        self.models = []
        for i, k in enumerate(self.k_values):
            print(f"Training KNN {i+1} mit k={k}...")
            knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
            knn.fit(X_train_scaled, y_train)
            self.models.append(knn)
            
        print("Training abgeschlossen!")
        
    def predict_single(self, X_test):
        """Macht Vorhersagen für Testdaten mit Details"""
        X_test_scaled = self.scaler.transform(X_test)
        
        # Vorhersagen aller 3 Modelle sammeln
        all_predictions = []
        for i, model in enumerate(self.models):
            pred = model.predict(X_test_scaled)
            all_predictions.append(pred)
            
        all_predictions = np.array(all_predictions)  # Shape: (3, n_samples)
        
        # Ensemble-Vorhersagen durch Mehrheitsvoting
        ensemble_predictions = []
        voting_details = []
        
        for i in range(len(X_test)):
            # Vorhersagen der 3 Modelle für Sample i
            votes = all_predictions[:, i]
            vote_counts = Counter(votes)
            
            # Prüfen ob mindestens 2 Modelle übereinstimmen
            max_votes = max(vote_counts.values())
            
            if max_votes >= 2:
                # Mehrheit gefunden - nehme den Wert mit den meisten Stimmen
                final_prediction = vote_counts.most_common(1)[0][0]
            else:
                # Keine Mehrheit - verwende Default
                final_prediction = self.default_prediction
                
            ensemble_predictions.append(final_prediction)
            voting_details.append({
                'votes': votes,
                'vote_counts': dict(vote_counts),
                'max_votes': max_votes,
                'final_prediction': final_prediction,
                'used_default': max_votes < 2
            })
            
        return np.array(ensemble_predictions), voting_details
    
    def predict(self, X_test):
        """Macht nur die Vorhersagen ohne Details"""
        predictions, _ = self.predict_single(X_test)
        return predictions
    
    def evaluate(self, X_test, y_test, show_details=True):
        """Evaluiert das Ensemble auf Testdaten"""
        predictions, voting_details = self.predict_single(X_test)
        
        # Metriken berechnen
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
        recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
        
        if show_details:
            print(f"\n=== KNN Ensemble Evaluation ===")
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            
            # Statistiken über das Voting
            default_used = sum(1 for detail in voting_details if detail['used_default'])
            majority_used = len(voting_details) - default_used
            
            print(f"\nVoting Statistiken:")
            print(f"Mehrheit erreicht: {majority_used}/{len(voting_details)} ({majority_used/len(voting_details)*100:.1f}%)")
            print(f"Default verwendet: {default_used}/{len(voting_details)} ({default_used/len(voting_details)*100:.1f}%)")
            
            # Classification Report
            print(f"\nClassification Report:")
            print(classification_report(y_test, predictions))
            
            # Confusion Matrix visualisieren
            self.plot_confusion_matrix(y_test, predictions)
            
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'voting_details': voting_details
        }
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plottet die Confusion Matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=sorted(np.unique(y_true)), 
                    yticklabels=sorted(np.unique(y_true)))
        plt.title('Confusion Matrix - KNN Ensemble')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    def analyze_individual_models(self, X_test, y_test):
        """Analysiert die Performance der einzelnen KNN-Modelle"""
        X_test_scaled = self.scaler.transform(X_test)
        
        print("\n=== Performance der einzelnen KNN-Modelle ===")
        print("-" * 60)
        
        individual_results = []
        for i, (model, k) in enumerate(zip(self.models, self.k_values)):
            pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, pred)
            prec = precision_score(y_test, pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, pred, average='weighted', zero_division=0)
            
            individual_results.append({
                'k': k,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1_score': f1
            })
            
            print(f"KNN {i+1} (k={k}): Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
            
        return individual_results

def add_sensor_feature(X):
    """Identifiziert Sensoren mittels K-Means Clustering"""
    km = KMeans(n_clusters=3, random_state=42)
    feature_13 = km.fit_predict(X)
    X_sensor = X.copy()
    X_sensor['sensor1'] = (feature_13 == 0).astype(int)
    X_sensor['sensor2'] = (feature_13 == 1).astype(int)
    X_sensor['sensor3'] = (feature_13 == 2).astype(int)
    return X_sensor

def calibrate_sensors(X_sensor):
    """Kalibriert die Sensoren durch Mittelwert-Normalisierung"""
    X_calibrated = X_sensor.copy()
    feature_cols = [col for col in X_sensor.columns if not col.startswith('sensor')]
    sensor_cols = ['sensor1', 'sensor2', 'sensor3']
    
    print("Kalibrierung der Sensoren:")
    print("-" * 50)
    
    for feature in feature_cols:
        overall_mean = X_sensor[feature].mean()
        print(f"\nFeature: {feature}")
        print(f"Gesamtmittelwert: {overall_mean:.4f}")
        
        for i, sensor_col in enumerate(sensor_cols):
            sensor_mask = X_sensor[sensor_col] == 1
            
            if sensor_mask.sum() > 0:
                sensor_mean = X_sensor.loc[sensor_mask, feature].mean()
                correction_factor = overall_mean - sensor_mean
                print(f"  Sensor {i+1}: Mittelwert = {sensor_mean:.4f}, Korrektur = {correction_factor:.4f}")
                X_calibrated.loc[sensor_mask, feature] += correction_factor
    
    return X_calibrated

def remove_sensor_columns(X_calibrated):
    """Entfernt die Sensor-Spalten nach der Kalibrierung"""
    sensor_cols = ['sensor1', 'sensor2', 'sensor3']
    X_final = X_calibrated.drop(columns=sensor_cols)
    return X_final

def create_visualizations(X_final, y):
    """Erstellt PCA, t-SNE und UMAP Visualisierungen"""
    print("\n=== Datenvisualisierung ===")
    
    # Standardisierung für Visualisierung
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_final)
    
    # PCA Visualization
    print("Erstelle PCA Visualisierung...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], 
                    hue=y, 
                    palette="Set1", 
                    s=15,
                    alpha=0.7,
                    edgecolor='none')
    plt.title(f"PCA – First Two Principal Components\n(Explained Variance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%})", 
            fontsize=14, fontweight='bold')
    plt.xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]:.1%} Variance)", fontsize=12)
    plt.ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]:.1%} Variance)", fontsize=12)
    plt.legend(title='Class', title_fontsize=12, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.show()

    # t-SNE Visualization
    print("Erstelle t-SNE Visualisierung...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=50, learning_rate=200, max_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], 
                hue=y, 
                palette="Set1", 
                s=15,
                alpha=0.7,
                edgecolor='none')
    plt.title("t-SNE Visualization", fontsize=14, fontweight='bold')
    plt.xlabel("t-SNE 1", fontsize=12)
    plt.ylabel("t-SNE 2", fontsize=12)
    plt.legend(title='Class', title_fontsize=12, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.show()

    # UMAP Visualization
    print("Erstelle UMAP Visualisierung...")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=X_umap[:,0], y=X_umap[:,1], 
                hue=y, 
                palette="Set1", 
                s=15,
                alpha=0.7,
                edgecolor='none')
    plt.title("UMAP Projection of Sensor Data", fontsize=14, fontweight='bold')
    plt.xlabel("UMAP-1", fontsize=12)
    plt.ylabel("UMAP-2", fontsize=12)
    plt.legend(title='Class', title_fontsize=12, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.show()

def predict_with_knn_ensemble(X_test, ensemble):
    """Macht Vorhersagen mit dem trainierten KNN Ensemble"""
    # Sensor-Spalten entfernen falls vorhanden
    feature_cols = [col for col in X_test.columns if not col.startswith('sensor')]
    X_test_features = X_test[feature_cols] if len(feature_cols) < len(X_test.columns) else X_test
    
    # Vorhersagen machen
    labels = ensemble.predict(X_test_features)
    
    # Für Outlier-Detektion: Einfacher Ansatz basierend auf Prediction Confidence
    # Hier nehmen wir an, dass Samples mit Default-Prediction eher Outlier sind
    predictions, voting_details = ensemble.predict_single(X_test_features)
    outliers = np.array([1 if detail['used_default'] else 0 for detail in voting_details])
    
    return labels, outliers

def generate_submission(test_data, ensemble):
    """Generiert Submission mit dem KNN Ensemble"""
    label_predictions, outlier_predictions = predict_with_knn_ensemble(test_data, ensemble)
    
    submission_df = pd.DataFrame({ 
        "id": test_data["id"],
        "label": label_predictions,
        "outlier": outlier_predictions
    })
    return submission_df

def train_outlier_gmm(X_train, outlier_fraction=0.2):
    gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=0)
    gmm.fit(X_train)
    log_probs = gmm.score_samples(X_train)
    threshold = np.percentile(log_probs, 100 * outlier_fraction)
    return gmm, threshold

def predict_outliers_gmm(gmm, threshold, X_new):
    log_probs_new = gmm.score_samples(X_new)
    return log_probs_new < threshold  # True = Outlier

def main():
    print("=== KNN Ensemble für Sensordaten ===")
    print("Lade Daten...")
    
    # Daten laden
    df = pd.read_csv("D.csv")
    X = df.iloc[:,1:13]  # Features
    y = df.iloc[:, -1]   # Labels
    
    print(f"Daten geladen: {X.shape[0]} Samples, {X.shape[1]} Features")
    print(f"Klassen: {sorted(np.unique(y))}")
    print(f"Klassenverteilung: {Counter(y)}")

    gmm, threshold = train_outlier_gmm(X, outlier_fraction=0.2)
    outlier_mask_train = predict_outliers_gmm(gmm, threshold, X)
    X = X[~outlier_mask_train]
    y = y[~outlier_mask_train]
    
    # 1. Sensoren identifizieren
    print("\n1. Identifiziere Sensoren...")
    X_sensor = add_sensor_feature(X)
    
    # Sensor-Verteilung anzeigen
    print("Sensor-Verteilung:")
    for i in range(1, 4):
        count = X_sensor[f'sensor{i}'].sum()
        print(f"  Sensor {i}: {count} Samples ({count/len(X_sensor)*100:.1f}%)")
    
    # 2. Sensoren kalibrieren
    print("\n2. Kalibriere Sensoren...")
    X_calibrated = calibrate_sensors(X_sensor)
    
    # 3. Sensor-Spalten entfernen für finale Features
    X_final = remove_sensor_columns(X_calibrated)
    print(f"\nFinale Feature-Matrix: {X_final.shape}")
    
    # 4. Visualisierungen erstellen
    create_visualizations(X_final, y)
    
    # 5. Train-Test Split
    print("\n5. Train-Test Split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training Set: {X_train.shape[0]} Samples")
    print(f"Test Set: {X_test.shape[0]} Samples")
    
    # 6. KNN Ensemble trainieren
    print("\n6. Trainiere KNN Ensemble...")
    ensemble = KNNEnsemble(
        k_values=[5, 7, 9],  # Verschiedene k-Werte
        default_prediction=1  # Default-Klasse
    )
    
    ensemble.fit(X_train, y_train)
    
    # 7. Modell evaluieren
    print("\n7. Evaluiere Modell...")
    results = ensemble.evaluate(X_test, y_test, show_details=True)
    
    # 8. Einzelne Modelle analysieren
    individual_results = ensemble.analyze_individual_models(X_test, y_test)
    
    # 9. Vergleich anzeigen
    ensemble_acc = results['accuracy']
    best_individual_acc = max([r['accuracy'] for r in individual_results])
    
    print(f"\n=== Gesamtvergleich ===")
    print(f"Ensemble Accuracy:     {ensemble_acc:.4f}")
    print(f"Bestes Einzelmodell:   {best_individual_acc:.4f}")
    print(f"Verbesserung:          {ensemble_acc - best_individual_acc:+.4f}")
    
    # # 10. Submissions erstellen
    # print("\n8. Erstelle Submissions...")
    
    # # Leaderboard Submission
    # try:
    #     df_leaderboard = pd.read_csv("D_test_leaderboard.csv")
    #     # Sensor-Features für Test-Daten hinzufügen und kalibrieren
    #     X_test_leader = df_leaderboard.iloc[:, 1:]  # Alle Features außer ID
    #     X_test_leader_sensor = add_sensor_feature(X_test_leader)
    #     X_test_leader_calibrated = calibrate_sensors(X_test_leader_sensor)
        
    #     submission_leader = generate_submission(df_leaderboard, ensemble)
    #     submission_leader.to_csv("submission_leaderboard_KNN_Ensemble.csv", index=False)
    #     print("Leaderboard Submission erstellt: submission_leaderboard_KNN_Ensemble.csv")
    # except FileNotFoundError:
    #     print("D_test_leaderboard.csv nicht gefunden - überspringe Leaderboard Submission")
    
    # # Final Submission
    # try:
    #     df_final = pd.read_csv("D_test_final.csv")
    #     X_test_final = df_final.iloc[:, 1:]
    #     X_test_final_sensor = add_sensor_feature(X_test_final)
    #     X_test_final_calibrated = calibrate_sensors(X_test_final_sensor)
        
    #     submission_final = generate_submission(df_final, ensemble)
    #     submission_final.to_csv("submission_final_KNN_Ensemble.csv", index=False)
    #     print("Final Submission erstellt: submission_final_KNN_Ensemble.csv")
    # except FileNotFoundError:
    #     print("D_test_final.csv nicht gefunden - überspringe Final Submission")
    
    # print("\n=== Fertig! ===")
    # return ensemble, results

if __name__ == "__main__":
    ensemble, results = main()