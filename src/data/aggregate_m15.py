"""
Phase 2 - Agregation M1 -> M15
Agrege les donnees 1 minute en bougies 15 minutes
Filtre les bougies incompletes (< 8 bougies M1)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


class M15Aggregator:
    """Agrege les donnees M1 en M15"""
    
    def __init__(self, raw_data_path: str = "data/raw", output_path: str = "data/processed"):
        self.raw_data_path = Path(raw_data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def load_m1(self, year: int) -> pd.DataFrame:
        """Charge les donnees M1 brutes"""
        file_path = self.raw_data_path / f"{year}.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier {file_path} introuvable")
        
        # Charger sans en-tete
        df = pd.read_csv(file_path, header=None)
        df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
        
        # Creer timestamp
        df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        
        # Trier et supprimer les doublons
        df = df.sort_values('timestamp').drop_duplicates(subset='timestamp', keep='first')
        df = df.reset_index(drop=True)
        
        print(f"M1 {year} charge: {len(df):,} lignes")
        
        return df
    
    def aggregate_to_m15(self, df_m1: pd.DataFrame, min_candles: int = 8) -> pd.DataFrame:
        """
        Agrege M1 en M15 selon les regles :
        - open_15m = open de la 1ere minute
        - high_15m = max(high) sur 15 minutes
        - low_15m = min(low) sur 15 minutes
        - close_15m = close de la derniere minute
        - volume_15m = sum(volume) sur 15 minutes
        
        Args:
            df_m1: DataFrame M1
            min_candles: Nombre minimum de bougies M1 pour valider une bougie M15 (defaut: 8)
        """
        
        # Definir l'index temporel
        df_m1 = df_m1.set_index('timestamp')
        
        # Agregation avec resample
        df_m15 = df_m1.resample('15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Compter le nombre de bougies M1 par periode M15
        count_m1 = df_m1.resample('15min').size()
        df_m15['n_candles_m1'] = count_m1
        
        # Filtrer : garder seulement les bougies avec au moins min_candles bougies M1
        print(f"  Avant filtrage: {len(df_m15):,} bougies M15")
        print(f"  Distribution du nombre de bougies M1 par periode M15:")
        print(f"    {df_m15['n_candles_m1'].value_counts().sort_index().head(20)}")
        
        df_m15_filtered = df_m15[df_m15['n_candles_m1'] >= min_candles].copy()
        
        print(f"  Apres filtrage (>= {min_candles} bougies M1): {len(df_m15_filtered):,} bougies M15")
        print(f"  Bougies supprimees: {len(df_m15) - len(df_m15_filtered):,} ({(len(df_m15) - len(df_m15_filtered))/len(df_m15)*100:.2f}%)")
        
        # Renommer les colonnes
        df_m15_filtered = df_m15_filtered.rename(columns={
            'open': 'open_15m',
            'high': 'high_15m',
            'low': 'low_15m',
            'close': 'close_15m',
            'volume': 'volume_15m'
        })
        
        # Supprimer les lignes avec valeurs manquantes dans OHLC
        df_m15_filtered = df_m15_filtered.dropna(subset=['open_15m', 'high_15m', 'low_15m', 'close_15m'])
        
        # Reset index pour avoir timestamp en colonne
        df_m15_filtered = df_m15_filtered.reset_index()
        df_m15_filtered.rename(columns={'timestamp': 'timestamp_15m'}, inplace=True)
        
        print(f"\nM15 agregees finales: {len(df_m15_filtered):,} bougies")
        print(f"Reduction: {len(df_m1)} -> {len(df_m15_filtered)} ({len(df_m15_filtered)/len(df_m1)*100:.2f}%)")
        
        return df_m15_filtered
    
    def validate_m15(self, df_m15: pd.DataFrame, year: int) -> Dict:
        """Valide les donnees M15 agregees"""
        
        print(f"\nValidation M15 - {year}")
        print("="*60)
        
        issues = []
        
        # 1. Verifier la regularite (15 minutes)
        time_diffs = df_m15['timestamp_15m'].diff()
        expected_diff = pd.Timedelta(minutes=15)
        
        irregular = time_diffs[time_diffs != expected_diff].dropna()
        if len(irregular) > 0:
            print(f"  Intervalles irreguliers: {len(irregular)}")
            issues.append(f"Intervalles irreguliers: {len(irregular)}")
        else:
            print(f"  Regularite: OK (toutes les 15 minutes)")
        
        # 2. Verifier les incoherences OHLC
        high_low_error = (df_m15['high_15m'] < df_m15['low_15m']).sum()
        if high_low_error > 0:
            print(f"  ERREUR: High < Low sur {high_low_error} bougies")
            issues.append(f"High < Low: {high_low_error}")
        else:
            print(f"  Coherence OHLC: OK")
        
        # 3. Verifier les prix negatifs
        neg_prices = (
            (df_m15['open_15m'] <= 0) | 
            (df_m15['high_15m'] <= 0) | 
            (df_m15['low_15m'] <= 0) | 
            (df_m15['close_15m'] <= 0)
        ).sum()
        
        if neg_prices > 0:
            print(f"  ERREUR: Prix negatifs/nuls sur {neg_prices} bougies")
            issues.append(f"Prix negatifs: {neg_prices}")
        else:
            print(f"  Prix positifs: OK")
        
        # 4. Valeurs manquantes
        missing = df_m15[['open_15m', 'high_15m', 'low_15m', 'close_15m']].isnull().sum().sum()
        if missing > 0:
            print(f"  ERREUR: {missing} valeurs manquantes")
            issues.append(f"Valeurs manquantes: {missing}")
        else:
            print(f"  Valeurs manquantes: OK")
        
        # 5. Statistiques
        print(f"\nStatistiques M15:")
        print(f"  Periode: {df_m15['timestamp_15m'].min()} -> {df_m15['timestamp_15m'].max()}")
        print(f"  Duree: {(df_m15['timestamp_15m'].max() - df_m15['timestamp_15m'].min()).days} jours")
        print(f"  Prix moyen: {df_m15['close_15m'].mean():.5f}")
        print(f"  Volatilite: {df_m15['close_15m'].std():.5f}")
        
        if 'n_candles_m1' in df_m15.columns:
            print(f"  Bougies M1 par M15 (moyenne): {df_m15['n_candles_m1'].mean():.2f}")
            print(f"  Bougies M1 par M15 (min): {df_m15['n_candles_m1'].min()}")
            print(f"  Bougies M1 par M15 (max): {df_m15['n_candles_m1'].max()}")
        
        print("="*60)
        
        if not issues:
            print("Validation: SUCCES")
        else:
            print(f"Validation: {len(issues)} probleme(s) detecte(s)")
        
        return {
            'year': year,
            'n_candles': len(df_m15),
            'issues': issues,
            'valid': len(issues) == 0
        }
    
    def save_m15(self, df_m15: pd.DataFrame, year: int):
        """Sauvegarde les donnees M15"""
        output_file = self.output_path / f"m15_{year}.csv"
        df_m15.to_csv(output_file, index=False)
        print(f"\nSauvegarde: {output_file}")
        print(f"Taille: {output_file.stat().st_size / (1024*1024):.2f} MB")
    
    def process_year(self, year: int, min_candles: int = 8) -> pd.DataFrame:
        """Pipeline complet pour une annee"""
        
        print(f"\n{'='*60}")
        print(f"TRAITEMENT {year}")
        print(f"{'='*60}\n")
        
        # 1. Charger M1
        df_m1 = self.load_m1(year)
        
        # 2. Agreger en M15 avec filtrage
        df_m15 = self.aggregate_to_m15(df_m1, min_candles=min_candles)
        
        # 3. Valider
        validation = self.validate_m15(df_m15, year)
        
        # 4. Sauvegarder
        self.save_m15(df_m15, year)
        
        return df_m15
    
    def process_all_years(self, min_candles: int = 8) -> Dict[int, pd.DataFrame]:
        """Traite toutes les annees"""
        
        results = {}
        
        print(f"\nCritere: Minimum {min_candles} bougies M1 par periode M15\n")
        
        for year in [2022, 2023, 2024]:
            try:
                df_m15 = self.process_year(year, min_candles=min_candles)
                results[year] = df_m15
            except Exception as e:
                print(f"\nERREUR {year}: {e}")
        
        return results


def main():
    """Fonction principale"""
    
    print("="*60)
    print("PHASE 2 - AGREGATION M1 -> M15")
    print("="*60)
    
    # Configurable : minimum 8 bougies M1 par bougie M15
    MIN_CANDLES = 8
    
    aggregator = M15Aggregator()
    results = aggregator.process_all_years(min_candles=MIN_CANDLES)
    
    print(f"\n{'='*60}")
    print("RESUME FINAL")
    print(f"{'='*60}\n")
    
    for year, df in results.items():
        print(f"{year}: {len(df):,} bougies M15 (>= {MIN_CANDLES} bougies M1 par periode)")
    
    print(f"\nTotal: {sum(len(df) for df in results.values()):,} bougies M15")
    print("\nFichiers sauvegardes dans: data/processed/")
    
    return results


if __name__ == "__main__":
    results = main()