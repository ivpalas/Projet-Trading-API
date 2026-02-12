"""
Feature Engineering for ML Models
Adds lag features, rolling statistics, and time-based features
"""

import pandas as pd
import numpy as np
from typing import List, Dict


class FeatureEngineer:
    """Create features for ML models from OHLCV + technical indicators data"""
    
    def __init__(self, config: Dict = None):
        """
        Args:
            config: Configuration dict with feature parameters
        """
        self.config = config or self._default_config()
    
    @staticmethod
    def _default_config() -> Dict:
        """Default configuration for feature engineering"""
        return {
            'lag_periods': [1, 2, 3, 5, 10, 20],  # Lag features
            'rolling_windows': [5, 10, 20, 50],   # Rolling statistics
            'price_columns': ['close', 'high', 'low'],
            'volume_lags': [1, 2, 5],
            'include_time_features': True,
        }
    
    def create_lag_features(self, df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """
        Create lag features for specified columns
        
        Args:
            df: Input dataframe
            columns: Columns to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features added
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, columns: List[str], windows: List[int]) -> pd.DataFrame:
        """
        Create rolling statistics (mean, std, min, max)
        
        Args:
            df: Input dataframe
            columns: Columns to compute rolling stats for
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features added
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            for window in windows:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window).std()
                df[f'{col}_rolling_min_{window}'] = df[col].rolling(window).min()
                df[f'{col}_rolling_max_{window}'] = df[col].rolling(window).max()
        
        return df
    
    def create_price_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create price change features (returns, log returns)
        
        Args:
            df: Input dataframe with 'close' column
            
        Returns:
            DataFrame with price change features
        """
        df = df.copy()
        
        if 'close' not in df.columns:
            return df
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Returns over different periods
        for period in [5, 10, 20]:
            df[f'returns_{period}'] = df['close'].pct_change(period)
        
        # Volatility (rolling std of returns)
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['returns'].rolling(window).std()
        
        return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from timestamp
        
        Args:
            df: Input dataframe with timestamp index
            
        Returns:
            DataFrame with time features
        """
        df = df.copy()
        
        if not isinstance(df.index, pd.DatetimeIndex):
            return df
        
        # Extract time components
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Trading session indicators (London/NY)
        df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
        df['overlap_session'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(int)
        
        return df
    
    def create_price_levels_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on price levels (distance from high/low, etc.)
        
        Args:
            df: Input dataframe with OHLC columns
            
        Returns:
            DataFrame with price level features
        """
        df = df.copy()
        
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return df
        
        # Intrabar price position
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        # Distance from high/low
        for window in [20, 50, 100]:
            df[f'dist_from_high_{window}'] = (df['high'].rolling(window).max() - df['close']) / df['close']
            df[f'dist_from_low_{window}'] = (df['close'] - df['low'].rolling(window).min()) / df['close']
        
        # Body and shadow sizes
        df['body_size'] = abs(df['close'] - df['open']) / df['close']
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
        
        return df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features at once
        
        Args:
            df: Input dataframe with OHLCV + technical indicators
            
        Returns:
            DataFrame with all features added
        """
        print("Creating ML features...")
        
        # 1. Lag features
        price_cols = [col for col in self.config['price_columns'] if col in df.columns]
        df = self.create_lag_features(df, price_cols, self.config['lag_periods'])
        
        if 'volume' in df.columns:
            df = self.create_lag_features(df, ['volume'], self.config['volume_lags'])
        
        # 2. Rolling statistics
        df = self.create_rolling_features(df, price_cols, self.config['rolling_windows'])
        
        # 3. Price changes
        df = self.create_price_change_features(df)
        
        # 4. Time features
        if self.config['include_time_features']:
            df = self.create_time_features(df)
        
        # 5. Price levels
        df = self.create_price_levels_features(df)
        
        # Drop NaN rows created by lags/rolling
        initial_rows = len(df)
        df = df.dropna()
        dropped_rows = initial_rows - len(df)
        
        print(f"✓ Created {len(df.columns)} features total")
        print(f"✓ Dropped {dropped_rows} rows with NaN values")
        print(f"✓ Final dataset: {len(df)} rows")
        
        return df


def add_target_variable(df: pd.DataFrame, threshold: float = 0.001, lookahead: int = 1) -> pd.DataFrame:
    """
    Add target variable for classification (UP/DOWN/HOLD)
    
    Args:
        df: Input dataframe with 'close' or 'close_15m' column
        threshold: Minimum price change to classify as UP/DOWN (default 0.1%)
        lookahead: How many periods ahead to look for target (default 1)
        
    Returns:
        DataFrame with 'target' column added
        
    Target values:
        1 = UP (price increases > threshold)
        0 = HOLD (price change within threshold)
        -1 = DOWN (price decreases > threshold)
    """
    df = df.copy()
    
    # Déterminer le nom de la colonne close
    close_col = 'close_15m' if 'close_15m' in df.columns else 'close'
    
    # Future price
    future_price = df[close_col].shift(-lookahead)
    
    # Price change
    price_change = (future_price - df[close_col]) / df[close_col]
    
    # Classify
    df['target'] = 0  # HOLD by default
    df.loc[price_change > threshold, 'target'] = 1   # UP
    df.loc[price_change < -threshold, 'target'] = -1  # DOWN
    
    # Remove rows where we can't compute target (end of dataset)
    df = df.dropna(subset=['target'])
    
    # Statistics
    target_counts = df['target'].value_counts().sort_index()
    print(f"\nTarget distribution:")
    print(f"  DOWN (-1): {target_counts.get(-1, 0)} ({target_counts.get(-1, 0)/len(df)*100:.1f}%)")
    print(f"  HOLD (0):  {target_counts.get(0, 0)} ({target_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"  UP (1):    {target_counts.get(1, 0)} ({target_counts.get(1, 0)/len(df)*100:.1f}%)")
    
    return df


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('/mnt/user-data/uploads')
    
    # Load data with technical indicators
    df = pd.read_parquet('data/processed/m15_features_2022.parquet')
    
    # Create features
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df)
    
    # Add target
    df_ml = add_target_variable(df_features, threshold=0.001)
    
    # Save
    df_ml.to_parquet('data/processed/ml_dataset_2022.parquet')
    print(f"\n✓ ML dataset saved: {df_ml.shape}")