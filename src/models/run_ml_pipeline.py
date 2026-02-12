"""
T07 - ML Models Pipeline
Complete pipeline: Feature Engineering → Training → Backtesting
"""

import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from feature_engineering import FeatureEngineer, add_target_variable
from ml_trainer import MLTrainer
from ml_backtester import MLBacktester, load_model_artifacts


def create_ml_datasets(years: list = [2022, 2023, 2024]):
    """
    Create ML datasets for multiple years
    
    Args:
        years: List of years to process
    """
    print("=" * 80)
    print("STEP 1: CREATE ML DATASETS")
    print("=" * 80)
    
    engineer = FeatureEngineer()
    
    for year in years:
        print(f"\n--- Processing {year} ---")
        
        # Load features (from T05)
        input_file = f'data/processed/m15_features_{year}.parquet'
        output_file = f'data/processed/ml_dataset_{year}.parquet'
        
        try:
            df = pd.read_parquet(input_file)
            print(f"✓ Loaded {len(df)} rows from {input_file}")
            
            # Create features
            df_features = engineer.create_all_features(df)
            
            # Add target
            df_ml = add_target_variable(df_features, threshold=0.001, lookahead=1)
            
            # Save
            df_ml.to_parquet(output_file)
            print(f"✓ Saved to {output_file} ({df_ml.shape})")
            
        except FileNotFoundError:
            print(f"✗ File not found: {input_file}")
            continue


def train_models(train_years: list = [2020, 2021, 2022]):
    """
    Train ML models on combined data from multiple years
    
    Args:
        train_years: Years to use for training
        
    Returns:
        MLTrainer instance with trained models
    """
    print("\n" + "=" * 80)
    print("STEP 2: TRAIN ML MODELS")
    print("=" * 80)
    
    # Load and combine training data
    dfs = []
    for year in train_years:
        file_path = f'data/processed/ml_dataset_{year}.parquet'
        try:
            df = pd.read_parquet(file_path)
            dfs.append(df)
            print(f"✓ Loaded {year}: {len(df)} rows")
        except FileNotFoundError:
            print(f"✗ File not found: {file_path}")
    
    if not dfs:
        print("No training data available!")
        return None
    
    # Combine
    df_train = pd.concat(dfs, ignore_index=False)
    print(f"\n✓ Combined training data: {len(df_train)} rows")
    
    # Initialize trainer
    trainer = MLTrainer(model_dir='models/saved')
    
    # Prepare data (80/20 split for validation)
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        df_train, 
        test_size=0.2,
        exclude_cols=['target']
    )
    
    # Train all models
    models = trainer.train_all_models(X_train, X_test, y_train, y_test)
    
    # Save models
    prefix = f"{'_'.join(map(str, train_years))}_"
    trainer.save_models(prefix=prefix)
    
    return trainer


def backtest_models(trainer: MLTrainer, test_year: int = 2023):
    """
    Backtest all trained models on test year
    
    Args:
        trainer: MLTrainer instance with trained models
        test_year: Year to use for backtesting
    """
    print("\n" + "=" * 80)
    print("STEP 3: BACKTEST ML MODELS")
    print("=" * 80)
    
    # Load test data
    test_file = f'data/processed/ml_dataset_{test_year}.parquet'
    try:
        df_test = pd.read_parquet(test_file)
        print(f"✓ Loaded test data ({test_year}): {len(df_test)} rows")
    except FileNotFoundError:
        print(f"✗ Test file not found: {test_file}")
        return
    
    # Initialize backtester
    backtester = MLBacktester(initial_capital=10000.0)
    
    # Backtest each model
    for model_name, model in trainer.models.items():
        results = backtester.backtest_ml_strategy(
            df_test,
            model,
            trainer.scalers['main'],
            trainer.feature_names,
            position_size=0.5,
            model_name=model_name
        )
    
    # Compare all models
    backtester.compare_models()


def main():
    """Run complete T07 pipeline"""
    print("\n" + "=" * 80)
    print("T07 - ML MODELS PIPELINE")
    print("=" * 80)
    
    # Configuration
    all_years = [2022, 2023, 2024]
    train_years = [2022, 2023]
    test_year = 2024
    
    # Step 1: Create ML datasets
    create_ml_datasets(years=all_years)
    
    # Step 2: Train models
    trainer = train_models(train_years=train_years)
    
    if trainer is None:
        print("\n✗ Training failed!")
        return
    
    # Step 3: Backtest models
    backtest_models(trainer, test_year=test_year)
    
    print("\n" + "=" * 80)
    print("✓ T07 PIPELINE COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()