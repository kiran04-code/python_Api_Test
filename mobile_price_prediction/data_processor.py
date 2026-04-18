import "dotenv/config";
"""
Mobile Price Prediction - Data Generation and Preprocessing Module
Generates synthetic mobile phone dataset and handles data preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

api_keys = "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"
class MobileDataGenerator:
    """Generate synthetic mobile phone dataset for price prediction"""
    
    def __init__(self, n_samples=2000):
        self.n_samples = n_samples
        np.random.seed(42)
    
    def generate_dataset(self):
        """Generate comprehensive mobile phone dataset"""
        
        data = {
            # Basic Features
            'battery_power': np.random.randint(500, 5000, self.n_samples),
            'blue': np.random.choice([0, 1], self.n_samples),
            'clock_speed': np.round(np.random.uniform(0.5, 3.0, self.n_samples), 1),
            'dual_sim': np.random.choice([0, 1], self.n_samples),
            'fc': np.random.randint(0, 20, self.n_samples),  # Front camera MP
            'four_g': np.random.choice([0, 1], self.n_samples),
            'int_memory': np.random.choice([16, 32, 64, 128, 256, 512], self.n_samples),
            'm_dep': np.round(np.random.uniform(0.1, 1.0, self.n_samples), 1),  # Mobile depth
            'mobile_wt': np.random.randint(80, 200, self.n_samples),
            'n_cores': np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], self.n_samples),
            'pc': np.random.randint(0, 20, self.n_samples),  # Primary camera MP
            'px_height': np.random.randint(0, 2000, self.n_samples),
            'px_width': np.random.randint(0, 2000, self.n_samples),
            'ram': np.random.randint(256, 8192, self.n_samples),
            'sc_h': np.random.randint(5, 20, self.n_samples),  # Screen height
            'sc_w': np.random.randint(3, 15, self.n_samples),  # Screen width
            'talk_time': np.random.randint(2, 20, self.n_samples),
            'three_g': np.random.choice([0, 1], self.n_samples),
            'touch_screen': np.random.choice([0, 1], self.n_samples),
            'wifi': np.random.choice([0, 1], self.n_samples),
            
            # Advanced Features
            'brand': np.random.choice(['Samsung', 'Apple', 'Xiaomi', 'OnePlus', 'Realme', 
                                      'Oppo', 'Vivo', 'Google', 'Nokia', 'Motorola'], self.n_samples),
            'os_type': np.random.choice(['Android', 'iOS', 'Other'], self.n_samples, p=[0.7, 0.2, 0.1]),
            'release_year': np.random.randint(2018, 2025, self.n_samples),
            'screen_size': np.round(np.random.uniform(4.5, 7.0, self.n_samples), 1),
            'refresh_rate': np.random.choice([60, 90, 120, 144], self.n_samples, p=[0.4, 0.3, 0.2, 0.1]),
            'fast_charging': np.random.choice([0, 1], self.n_samples, p=[0.4, 0.6]),
            'nfc': np.random.choice([0, 1], self.n_samples),
            'fingerprint': np.random.choice([0, 1], self.n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Calculate resolution in megapixels
        df['resolution_mp'] = (df['px_height'] * df['px_width']) / 1_000_000
        
        # Calculate screen resolution
        df['screen_resolution'] = df['px_height'] * df['px_width']
        
        # Generate realistic price based on features
        df['price'] = self._calculate_price(df)
        
        # Price range category (0-3)
        df['price_range'] = pd.cut(df['price'], 
                                   bins=[0, 10000, 20000, 40000, 100000],
                                   labels=[0, 1, 2, 3]).astype(int)
        
        return df
    
    def _calculate_price(self, df):
        """Calculate realistic mobile price based on specifications"""
        
        price = np.zeros(len(df))
        
        # Base price
        price += 5000
        
        # RAM impact (major factor)
        price += df['ram'] * 3.5
        
        # Storage impact
        price += df['int_memory'] * 15
        
        # Camera quality
        price += df['pc'] * 200  # Primary camera
        price += df['fc'] * 150  # Front camera
        
        # Battery
        price += df['battery_power'] * 1.2
        
        # Brand premium
        brand_premium = {
            'Apple': 25000,
            'Samsung': 8000,
            'Google': 10000,
            'OnePlus': 6000,
            'Xiaomi': 2000,
            'Realme': 1500,
            'Oppo': 3000,
            'Vivo': 3000,
            'Nokia': 2000,
            'Motorola': 2500
        }
        price += df['brand'].map(brand_premium)
        
        # OS premium
        price += df['os_type'].apply(lambda x: 15000 if x == 'iOS' else 0)
        
        # Features
        price += df['four_g'] * 2000
        price += df['nfc'] * 1500
        price += df['fingerprint'] * 1000
        price += df['fast_charging'] * 2000
        
        # Display quality
        price += df['refresh_rate'] * 30
        price += df['screen_size'] * 1000
        
        # Resolution
        price += df['resolution_mp'] * 500
        
        # Release year (newer = expensive)
        price += (df['release_year'] - 2018) * 2000
        
        # Add some noise
        noise = np.random.normal(0, 2000, len(df))
        price += noise
        
        # Ensure minimum price
        price = np.maximum(price, 3000)
        
        return np.round(price).astype(int)
    
    def save_dataset(self, df, filepath='data/mobile_data.csv'):
        """Save dataset to CSV"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"✓ Dataset saved to {filepath}")
        print(f"✓ Total samples: {len(df)}")
        print(f"✓ Features: {df.shape[1] - 2}")  # Exclude price and price_range
        return filepath


class DataPreprocessor:
    """Handle data preprocessing for ML models"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
    
    def preprocess(self, df, fit=True):
        """
        Preprocess the dataset
        Args:
            df: DataFrame with mobile data
            fit: Whether to fit scalers/encoders (True for training, False for prediction)
        """
        df_processed = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['brand', 'os_type']
        
        for col in categorical_cols:
            if col in df_processed.columns:
                if fit:
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col])
                    self.label_encoders[col] = le
                else:
                    le = self.label_encoders.get(col)
                    if le:
                        # Handle unseen labels
                        df_processed[col] = df_processed[col].apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
        
        # Define feature columns (exclude target variables)
        if fit:
            self.feature_columns = [col for col in df_processed.columns 
                                   if col not in ['price', 'price_range']]
        
        # Separate features and target
        X = df_processed[self.feature_columns]
        y_price = df_processed['price'] if 'price' in df_processed.columns else None
        y_range = df_processed['price_range'] if 'price_range' in df_processed.columns else None
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y_price, y_range
    
    def split_data(self, X, y_price, y_range, test_size=0.2, val_size=0.1):
        """Split data into train, validation, and test sets"""
        
        # First split: train+val vs test
        X_train_val, X_test, y_price_train_val, y_price_test, y_range_train_val, y_range_test = \
            train_test_split(X, y_price, y_range, test_size=test_size, random_state=42)
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_price_train, y_price_val, y_range_train, y_range_val = \
            train_test_split(X_train_val, y_price_train_val, y_range_train_val, 
                           test_size=val_size_adjusted, random_state=42)
        
        return {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_price_train': y_price_train, 'y_price_val': y_price_val, 'y_price_test': y_price_test,
            'y_range_train': y_range_train, 'y_range_val': y_range_val, 'y_range_test': y_range_test
        }
    
    def get_feature_names(self):
        """Get feature column names"""
        return self.feature_columns


if __name__ == "__main__":
    print("=" * 60)
    print("Mobile Price Prediction - Data Generation")
    print("=" * 60)
    
    # Generate dataset
    generator = MobileDataGenerator(n_samples=2000)
    df = generator.generate_dataset()
    
    # Save dataset
    generator.save_dataset(df)
    
    # Display sample
    print("\n" + "=" * 60)
    print("Sample Data:")
    print("=" * 60)
    print(df[['brand', 'ram', 'int_memory', 'pc', 'fc', 'battery_power', 'price']].head(10))
    
    print("\n" + "=" * 60)
    print("Price Distribution:")
    print("=" * 60)
    print(df['price_range'].value_counts().sort_index())
    print(f"\nPrice Range: ₹{df['price'].min():,} - ₹{df['price'].max():,}")
    print(f"Average Price: ₹{df['price'].mean():,.2f}")


