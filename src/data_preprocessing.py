"""
Data preprocessing module for Global School Electricity Access Data
Handles: cleaning, missing values, encoding, normalization, train/test split
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict


class DataPreprocessor:
    """Handles all data preprocessing operations"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.target_name = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file"""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with multiple strategies"""
        print("\nHandling missing values...")
        
        # Show missing values info
        missing_info = df.isnull().sum()
        if missing_info.sum() > 0:
            print(f"Missing values:\n{missing_info[missing_info > 0]}")
        
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        print(f"Missing values after handling: {df.isnull().sum().sum()}")
        return df
    
    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables using LabelEncoder"""
        print("\nEncoding categorical variables...")
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col == self.target_name:  # Skip target if it's categorical
                continue
                
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                if col in self.label_encoders:
                    df[col] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary target variable for electricity access"""
        print("\nCreating target variable...")
        
        # Assuming we want to classify schools with good electricity access
        # This logic depends on your actual data - adjust as needed
        if 'electricity_access_pct' in df.columns:
            df['target'] = (df['electricity_access_pct'] > df['electricity_access_pct'].median()).astype(int)
            self.target_name = 'target'
        elif 'access' in df.columns:
            df['target'] = (df['access'] > 0).astype(int)
            self.target_name = 'target'
        else:
            # Default: use first numeric column as proxy
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                df['target'] = (df[col] > df[col].median()).astype(int)
                self.target_name = 'target'
        
        print(f"Target variable distribution:\n{df['target'].value_counts()}")
        return df
    
    def normalize_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None, 
                          fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize features using StandardScaler"""
        print("\nNormalizing features...")
        
        if fit:
            X_train_scaled = self.scaler.fit_transform(X_train)
            print(f"Scaler fitted on training data")
        else:
            X_train_scaled = self.scaler.transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled, None
    
    def preprocess_pipeline(self, filepath: str, test_size: float = 0.2) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """Complete preprocessing pipeline"""
        print("=" * 60)
        print("STARTING PREPROCESSING PIPELINE")
        print("=" * 60)
        
        # Load
        df = self.load_data(filepath)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Create target
        df = self.create_target_variable(df)
        
        # Encode categorical
        df = self.encode_categorical(df, fit=True)
        
        # Separate features and target
        y = df[self.target_name].values
        X = df.drop(columns=[self.target_name])
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"\nTrain set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Normalize
        X_train_scaled, X_test_scaled = self.normalize_features(X_train, X_test, fit=True)
        
        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE")
        print("=" * 60)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, pd.DataFrame(X_train, columns=X.columns)


def get_preprocessor() -> DataPreprocessor:
    """Factory function to create preprocessor"""
    return DataPreprocessor(random_state=42)
