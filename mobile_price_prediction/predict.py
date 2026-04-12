"""
Mobile Price Prediction - Prediction Script
Load trained model and make predictions
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
from data_processor import DataPreprocessor

api_key = "uMhmjknx9QeGIgpyQdLa3VekNmPc"
class MobilePricePredictor:
    """Load trained model and predict mobile prices"""
    
    def __init__(self, model_path='models/mobile_price_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self):
        """Load trained model from file"""
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Please train the model first using train_model.py"
            )
        
        print(f"✓ Loading model from {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.preprocessor = model_data['preprocessor']
        self.feature_names = model_data['feature_names']
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Model type: {type(self.model).__name__}")
        print(f"✓ Features: {len(self.feature_names)}")
    
    def predict_single(self, mobile_specs):
        """
        Predict price for a single mobile phone
        Args:
            mobile_specs: Dictionary with mobile specifications
        Returns:
            Dictionary with prediction results
        """
        
        # Convert to DataFrame
        df = pd.DataFrame([mobile_specs])
        
        # Preprocess
        X, _, _ = self.preprocessor.preprocess(df, fit=False)
        
        # Predict
        predicted_price = self.model.predict(X)[0]
        
        # Determine price range
        if predicted_price < 10000:
            price_range = 0
            range_label = "Budget (< ₹10,000)"
        elif predicted_price < 20000:
            price_range = 1
            range_label = "Mid-Range (₹10,000 - ₹20,000)"
        elif predicted_price < 40000:
            price_range = 2
            range_label = "Premium (₹20,000 - ₹40,000)"
        else:
            price_range = 3
            range_label = "Flagship (> ₹40,000)"
        
        return {
            'predicted_price': round(predicted_price, 2),
            'price_range': price_range,
            'price_range_label': range_label,
            'formatted_price': f"₹{predicted_price:,.2f}"
        }
    
    def predict_batch(self, df):
        """
        Predict prices for multiple mobiles
        Args:
            df: DataFrame with mobile specifications
        Returns:
            DataFrame with predictions
        """
        
        # Preprocess
        X, _, _ = self.preprocessor.preprocess(df, fit=False)
        
        # Predict
        predictions = self.model.predict(X)
        
        # Add to DataFrame
        df_result = df.copy()
        df_result['predicted_price'] = np.round(predictions, 2)
        df_result['formatted_price'] = df_result['predicted_price'].apply(
            lambda x: f"₹{x:,.2f}"
        )
        
        return df_result
    
    def get_example_specs(self):
        """Get example mobile specifications for testing"""
        
        examples = {
            'Budget Phone': {
                'battery_power': 3000,
                'blue': 1,
                'clock_speed': 1.8,
                'dual_sim': 1,
                'fc': 5,
                'four_g': 1,
                'int_memory': 32,
                'm_dep': 0.6,
                'mobile_wt': 150,
                'n_cores': 4,
                'pc': 13,
                'px_height': 720,
                'px_width': 1280,
                'ram': 2048,
                'sc_h': 12,
                'sc_w': 7,
                'talk_time': 10,
                'three_g': 1,
                'touch_screen': 1,
                'wifi': 1,
                'brand': 'Xiaomi',
                'os_type': 'Android',
                'release_year': 2022,
                'screen_size': 6.1,
                'refresh_rate': 60,
                'fast_charging': 0,
                'nfc': 0,
                'fingerprint': 1
            },
            'Mid-Range Phone': {
                'battery_power': 4500,
                'blue': 1,
                'clock_speed': 2.4,
                'dual_sim': 1,
                'fc': 16,
                'four_g': 1,
                'int_memory': 128,
                'm_dep': 0.8,
                'mobile_wt': 180,
                'n_cores': 8,
                'pc': 48,
                'px_height': 1080,
                'px_width': 2340,
                'ram': 6144,
                'sc_h': 16,
                'sc_w': 7,
                'talk_time': 15,
                'three_g': 1,
                'touch_screen': 1,
                'wifi': 1,
                'brand': 'OnePlus',
                'os_type': 'Android',
                'release_year': 2023,
                'screen_size': 6.5,
                'refresh_rate': 120,
                'fast_charging': 1,
                'nfc': 1,
                'fingerprint': 1
            },
            'Flagship Phone': {
                'battery_power': 5000,
                'blue': 1,
                'clock_speed': 2.8,
                'dual_sim': 1,
                'fc': 32,
                'four_g': 1,
                'int_memory': 256,
                'm_dep': 0.9,
                'mobile_wt': 200,
                'n_cores': 8,
                'pc': 108,
                'px_height': 1440,
                'px_width': 3200,
                'ram': 8192,
                'sc_h': 18,
                'sc_w': 8,
                'talk_time': 18,
                'three_g': 1,
                'touch_screen': 1,
                'wifi': 1,
                'brand': 'Apple',
                'os_type': 'iOS',
                'release_year': 2024,
                'screen_size': 6.7,
                'refresh_rate': 120,
                'fast_charging': 1,
                'nfc': 1,
                'fingerprint': 1
            }
        }
        
        return examples


def interactive_prediction():
    """Interactive prediction mode"""
    
    print("=" * 60)
    print("Mobile Price Prediction - Interactive Mode")
    print("=" * 60)
    
    # Initialize predictor
    predictor = MobilePricePredictor()
    
    # Get example specs
    examples = predictor.get_example_specs()
    
    print("\n📱 Example Mobile Configurations:")
    print("-" * 60)
    for i, (name, specs) in enumerate(examples.items(), 1):
        print(f"{i}. {name}")
    
    print(f"\n4. Custom Input")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == '1':
        specs = examples['Budget Phone']
    elif choice == '2':
        specs = examples['Mid-Range Phone']
    elif choice == '3':
        specs = examples['Flagship Phone']
    elif choice == '4':
        print("\n" + "=" * 60)
        print("Enter Mobile Specifications")
        print("=" * 60)
        
        specs = {}
        specs['battery_power'] = int(input("Battery Power (mAh) [500-5000]: ") or 4000)
        specs['blue'] = int(input("Bluetooth (0/1): ") or 1)
        specs['clock_speed'] = float(input("Clock Speed (GHz) [0.5-3.0]: ") or 2.0)
        specs['dual_sim'] = int(input("Dual SIM (0/1): ") or 1)
        specs['fc'] = int(input("Front Camera (MP) [0-20]: ") or 8)
        specs['four_g'] = int(input("4G Support (0/1): ") or 1)
        specs['int_memory'] = int(input("Internal Memory (GB) [16/32/64/128/256/512]: ") or 128)
        specs['m_dep'] = float(input("Mobile Depth (cm) [0.1-1.0]: ") or 0.7)
        specs['mobile_wt'] = int(input("Mobile Weight (g) [80-200]: ") or 170)
        specs['n_cores'] = int(input("Number of Cores [1-8]: ") or 8)
        specs['pc'] = int(input("Primary Camera (MP) [0-20]: ") or 48)
        specs['px_height'] = int(input("Pixel Height [0-2000]: ") or 1080)
        specs['px_width'] = int(input("Pixel Width [0-2000]: ") or 2340)
        specs['ram'] = int(input("RAM (MB) [256-8192]: ") or 6144)
        specs['sc_h'] = int(input("Screen Height (cm) [5-20]: ") or 16)
        specs['sc_w'] = int(input("Screen Width (cm) [3-15]: ") or 7)
        specs['talk_time'] = int(input("Talk Time (hours) [2-20]: ") or 15)
        specs['three_g'] = int(input("3G Support (0/1): ") or 1)
        specs['touch_screen'] = int(input("Touch Screen (0/1): ") or 1)
        specs['wifi'] = int(input("WiFi (0/1): ") or 1)
        specs['brand'] = input("Brand [Samsung/Apple/Xiaomi/OnePlus/Realme/Oppo/Vivo/Google/Nokia/Motorola]: ") or 'Samsung'
        specs['os_type'] = input("OS Type [Android/iOS/Other]: ") or 'Android'
        specs['release_year'] = int(input("Release Year [2018-2024]: ") or 2023)
        specs['screen_size'] = float(input("Screen Size (inches) [4.5-7.0]: ") or 6.5)
        specs['refresh_rate'] = int(input("Refresh Rate (Hz) [60/90/120/144]: ") or 120)
        specs['fast_charging'] = int(input("Fast Charging (0/1): ") or 1)
        specs['nfc'] = int(input("NFC (0/1): ") or 1)
        specs['fingerprint'] = int(input("Fingerprint (0/1): ") or 1)
    else:
        print("❌ Invalid choice!")
        return
    
    # Make prediction
    print("\n" + "=" * 60)
    print("Prediction Result")
    print("=" * 60)
    
    result = predictor.predict_single(specs)
    
    print(f"\n💰 Predicted Price: {result['formatted_price']}")
    print(f"📊 Price Range: {result['price_range_label']}")
    print(f"🏷️ Category: {'Budget' if result['price_range'] == 0 else 'Mid-Range' if result['price_range'] == 1 else 'Premium' if result['price_range'] == 2 else 'Flagship'}")


def batch_prediction():
    """Batch prediction from CSV file"""
    
    print("=" * 60)
    print("Mobile Price Prediction - Batch Mode")
    print("=" * 60)
    
    input_file = input("\nEnter input CSV file path: ").strip()
    
    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        return
    
    # Load data
    df = pd.read_csv(input_file)
    print(f"✓ Loaded {len(df)} mobiles from {input_file}")
    
    # Initialize predictor
    predictor = MobilePricePredictor()
    
    # Predict
    df_result = predictor.predict_batch(df)
    
    # Save results
    output_file = input_file.replace('.csv', '_predictions.csv')
    df_result.to_csv(output_file, index=False)
    
    print(f"\n✓ Predictions saved to {output_file}")
    print(f"\n📊 Sample Predictions:")
    print(df_result[['brand', 'ram', 'int_memory', 'predicted_price']].head(10))


if __name__ == "__main__":
    print("=" * 60)
    print("Mobile Price Prediction System")
    print("=" * 60)
    
    print("\nSelect Mode:")
    print("1. Interactive Prediction (Single Mobile)")
    print("2. Batch Prediction (CSV File)")
    
    choice = input("\nEnter choice (1-2): ").strip()
    
    if choice == '1':
        interactive_prediction()
    elif choice == '2':
        batch_prediction()
    else:
        print("❌ Invalid choice!")

