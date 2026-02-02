"""
Machine Learning Classification Models Training
This module trains and evaluates 6 different classification models on the dataset
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, 
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Train and evaluate multiple classification models"""
    
    def __init__(self, data_path='data/breast_cancer.csv'):
        """Initialize the trainer with data"""
        self.data_path = data_path
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model_dir = 'models'
        
        # Create models directory if it doesn't exist
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
    
    def load_data(self):
        """Load and split the dataset"""
        df = pd.read_csv(self.data_path)
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        # Save scaler
        with open(f'{self.model_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Data loaded successfully!")
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
        print(f"Number of features: {self.X_train.shape[1]}")
        print(f"Class distribution (train): {np.bincount(self.y_train)}")
        print(f"Class distribution (test): {np.bincount(self.y_test)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """Train all 6 classification models"""
        print("\n" + "="*60)
        print("Training Classification Models...")
        print("="*60)
        
        # 1. Logistic Regression
        print("\n1. Training Logistic Regression...")
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = lr
        self._save_model(lr, 'logistic_regression')
        
        # 2. Decision Tree Classifier
        print("2. Training Decision Tree Classifier...")
        dt = DecisionTreeClassifier(random_state=42, max_depth=15)
        dt.fit(self.X_train, self.y_train)
        self.models['Decision Tree'] = dt
        self._save_model(dt, 'decision_tree')
        
        # 3. K-Nearest Neighbors
        print("3. Training K-Nearest Neighbors...")
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(self.X_train, self.y_train)
        self.models['K-Nearest Neighbor'] = knn
        self._save_model(knn, 'knn')
        
        # 4. Naive Bayes (Gaussian)
        print("4. Training Naive Bayes Classifier...")
        nb = GaussianNB()
        nb.fit(self.X_train, self.y_train)
        self.models['Naive Bayes'] = nb
        self._save_model(nb, 'naive_bayes')
        
        # 5. Random Forest
        print("5. Training Random Forest Classifier...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf
        self._save_model(rf, 'random_forest')
        
        # 6. XGBoost
        print("6. Training XGBoost Classifier...")
        xgb = XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
        xgb.fit(self.X_train, self.y_train)
        self.models['XGBoost'] = xgb
        self._save_model(xgb, 'xgboost')
        
        print("\nAll models trained successfully!")
    
    def evaluate_models(self):
        """Evaluate all models and compute metrics"""
        print("\n" + "="*60)
        print("Evaluating Models...")
        print("="*60)
        
        evaluation_metrics = {}
        
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name}...")
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'Model': model_name,
                'Accuracy': round(accuracy_score(self.y_test, y_pred), 4),
                'AUC': round(roc_auc_score(self.y_test, y_pred_proba), 4),
                'Precision': round(precision_score(self.y_test, y_pred), 4),
                'Recall': round(recall_score(self.y_test, y_pred), 4),
                'F1 Score': round(f1_score(self.y_test, y_pred), 4),
                'MCC': round(matthews_corrcoef(self.y_test, y_pred), 4),
            }
            
            # Store confusion matrix and classification report
            metrics['Confusion Matrix'] = confusion_matrix(self.y_test, y_pred)
            metrics['Classification Report'] = classification_report(self.y_test, y_pred, output_dict=True)
            
            evaluation_metrics[model_name] = metrics
            
            # Print metrics
            print(f"  Accuracy:  {metrics['Accuracy']}")
            print(f"  AUC:       {metrics['AUC']}")
            print(f"  Precision: {metrics['Precision']}")
            print(f"  Recall:    {metrics['Recall']}")
            print(f"  F1 Score:  {metrics['F1 Score']}")
            print(f"  MCC:       {metrics['MCC']}")
        
        # Save results
        self.results = evaluation_metrics
        self._save_results(evaluation_metrics)
        
        return evaluation_metrics
    
    def get_results_dataframe(self):
        """Get evaluation results as a pandas DataFrame"""
        results_list = []
        for model_name, metrics in self.results.items():
            result = {
                'Model': model_name,
                'Accuracy': metrics['Accuracy'],
                'AUC': metrics['AUC'],
                'Precision': metrics['Precision'],
                'Recall': metrics['Recall'],
                'F1 Score': metrics['F1 Score'],
                'MCC': metrics['MCC'],
            }
            results_list.append(result)
        
        return pd.DataFrame(results_list)
    
    def _save_model(self, model, model_name):
        """Save trained model to disk"""
        model_path = f'{self.model_dir}/{model_name}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"  Saved to {model_path}")
    
    def _save_results(self, results):
        """Save evaluation results"""
        results_path = f'{self.model_dir}/results.pkl'
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nResults saved to {results_path}")


def main():
    """Main execution function"""
    # Initialize trainer
    trainer = ModelTrainer(data_path='data/breast_cancer.csv')
    
    # Load data
    trainer.load_data()
    
    # Train models
    trainer.train_models()
    
    # Evaluate models
    results = trainer.evaluate_models()
    
    # Display results table
    print("\n" + "="*80)
    print("EVALUATION RESULTS SUMMARY")
    print("="*80)
    print(trainer.get_results_dataframe().to_string(index=False))
    
    # Save results DataFrame
    trainer.get_results_dataframe().to_csv(f'{trainer.model_dir}/evaluation_results.csv', index=False)
    print(f"\nResults CSV saved to {trainer.model_dir}/evaluation_results.csv")


if __name__ == "__main__":
    main()
