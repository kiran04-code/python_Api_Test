"""
Mobile Price Prediction - Model Training Module
Trains multiple ML models and selects the best one
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, 
    accuracy_score, classification_report, confusion_matrix
)
from data_processor import MobileDataGenerator, DataPreprocessor

api_key = "Awgbywsusbqiniqmsqosqp"
class MobilePriceTrainer:
    """Train and evaluate ML models for mobile price prediction"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.preprocessor = DataPreprocessor()
    
    def load_or_generate_data(self, data_path='data/mobile_data.csv', n_samples=2000):
        """Load existing data or generate new dataset"""
        
        if os.path.exists(data_path):
            print(f"✓ Loading existing dataset from {data_path}")
            df = pd.read_csv(data_path)
        else:
            print("⚠ Dataset not found. Generating new dataset...")
            generator = MobileDataGenerator(n_samples=n_samples)
            df = generator.generate_dataset()
            generator.save_dataset(df, data_path)
        
        print(f"✓ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
        return df
    
    def prepare_data(self, df):
        """Preprocess and split data"""
        
        print("\n" + "=" * 60)
        print("Data Preprocessing")
        print("=" * 60)
        
        # Preprocess
        X, y_price, y_range = self.preprocessor.preprocess(df, fit=True)
        
        # Split data
        data_splits = self.preprocessor.split_data(X, y_price, y_range)
        
        print(f"✓ Training set: {len(data_splits['X_train'])} samples")
        print(f"✓ Validation set: {len(data_splits['X_val'])} samples")
        print(f"✓ Test set: {len(data_splits['X_test'])} samples")
        
        return data_splits
    
    def train_models(self, data_splits):
        """Train multiple ML models"""
        
        print("\n" + "=" * 60)
        print("Model Training")
        print("=" * 60)
        
        X_train = data_splits['X_train']
        y_price_train = data_splits['y_price_train']
        y_range_train = data_splits['y_range_train']
        
        # Define models for price prediction (regression)
        regressors = {
            'RandomForest': RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'Ridge': Ridge(alpha=1.0, random_state=42),
            'Lasso': Lasso(alpha=0.1, random_state=42),
            'LinearRegression': LinearRegression()
        }
        
        # Train regression models (predict actual price)
        for name, model in regressors.items():
            print(f"\n📊 Training {name}...")
            model.fit(X_train, y_price_train)
            self.models[f'{name}_Regression'] = model
            
            # Evaluate on validation set
            y_pred = model.predict(data_splits['X_val'])
            mae = mean_absolute_error(data_splits['y_price_val'], y_pred)
            rmse = np.sqrt(mean_squared_error(data_splits['y_price_val'], y_pred))
            r2 = r2_score(data_splits['y_price_val'], y_pred)
            
            self.results[f'{name}_Regression'] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2
            }
            
            print(f"  ✓ MAE: ₹{mae:,.2f}")
            print(f"  ✓ RMSE: ₹{rmse:,.2f}")
            print(f"  ✓ R² Score: {r2:.4f}")
        
        # Train classification model (predict price range)
        print(f"\n📊 Training RandomForest Classification...")
        rf_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        rf_classifier.fit(X_train, y_range_train)
        self.models['RandomForest_Classification'] = rf_classifier
        
        # Evaluate classification
        y_range_pred = rf_classifier.predict(data_splits['X_val'])
        accuracy = accuracy_score(data_splits['y_range_val'], y_range_pred)
        
        self.results['RandomForest_Classification'] = {
            'Accuracy': accuracy,
            'Classification_Report': classification_report(
                data_splits['y_range_val'], 
                y_range_pred,
                output_dict=True
            )
        }
        
        print(f"  ✓ Accuracy: {accuracy:.4f}")
        print(f"  ✓ Classification Report Generated")
        
        return self.results
    
    def evaluate_best_model(self, data_splits):
        """Evaluate the best model on test set"""
        
        print("\n" + "=" * 60)
        print("Finding Best Model")
        print("=" * 60)
        
        # Find best regression model based on R² score
        best_model_name = None
        best_r2 = -float('inf')
        
        for name, metrics in self.results.items():
            if 'Regression' in name and 'R2' in metrics:
                if metrics['R2'] > best_r2:
                    best_r2 = metrics['R2']
                    best_model_name = name
        
        if best_model_name:
            self.best_model = self.models[best_model_name]
            
            # Test on test set
            y_pred = self.best_model.predict(data_splits['X_test'])
            mae = mean_absolute_error(data_splits['y_price_test'], y_pred)
            rmse = np.sqrt(mean_squared_error(data_splits['y_price_test'], y_pred))
            r2 = r2_score(data_splits['y_price_test'], y_pred)
            
            print(f"\n🏆 Best Model: {best_model_name}")
            print(f"   MAE: ₹{mae:,.2f}")
            print(f"   RMSE: ₹{rmse:,.2f}")
            print(f"   R² Score: {r2:.4f}")
            
            # Sample predictions
            print("\n📋 Sample Predictions:")
            sample_indices = np.random.choice(len(data_splits['X_test']), 5, replace=False)
            for idx in sample_indices:
                actual = data_splits['y_price_test'].iloc[idx]
                predicted = y_pred[idx]
                error = abs(actual - predicted)
                print(f"   Actual: ₹{actual:,.0f} | Predicted: ₹{predicted:,.0f} | Error: ₹{error:,.0f}")
        
        return best_model_name
    
    def save_model(self, model_path='models/mobile_price_model.pkl', 
                   metadata_path='models/model_metadata.json'):
        """Save trained model and metadata"""
        
        print("\n" + "=" * 60)
        print("Saving Model")
        print("=" * 60)
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model and preprocessor
        model_data = {
            'model': self.best_model,
            'preprocessor': self.preprocessor,
            'feature_names': self.preprocessor.get_feature_names()
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Model saved to {model_path}")
        
        # Save metadata
        metadata = {
            'model_type': type(self.best_model).__name__,
            'feature_count': len(self.preprocessor.get_feature_names()),
            'features': self.preprocessor.get_feature_names(),
            'results': {k: {mk: mv for mk, mv in v.items() if mk != 'Classification_Report'} 
                       for k, v in self.results.items()}
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"✓ Metadata saved to {metadata_path}")
        
        return model_path
    
    def feature_importance(self, top_n=15):
        """Display feature importance from best model"""
        
        print("\n" + "=" * 60)
        print("Feature Importance (Top 15)")
        print("=" * 60)
        
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            feature_names = self.preprocessor.get_feature_names()
            
            # Sort by importance
            indices = np.argsort(importances)[::-1]
            
            print(f"\n{'Feature':<25} {'Importance':>12}")
            print("-" * 40)
            
            for i in range(min(top_n, len(feature_names))):
                idx = indices[i]
                print(f"{feature_names[idx]:<25} {importances[idx]:>12.4f}")
        else:
            print("⚠ Model doesn't support feature importance")


def main():
    """Main training pipeline"""
    
    print("=" * 60)
    print("Mobile Price Prediction - Model Training")
    print("=" * 60)
    
    # Initialize trainer
    trainer = MobilePriceTrainer()
    
    # Load/generate data
    df = trainer.load_or_generate_data()
    
    # Prepare data
    data_splits = trainer.prepare_data(df)
    
    # Train models
    results = trainer.train_models(data_splits)
    
    # Evaluate best model
    best_model = trainer.evaluate_best_model(data_splits)
    
    # Feature importance
    trainer.feature_importance()
    
    # Save model
    trainer.save_model()
    
    print("\n" + "=" * 60)
    print("✓ Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
