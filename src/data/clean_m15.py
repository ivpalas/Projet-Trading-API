"""
Phase 3 - Nettoyage M15
Controle final des donnees M15 :
- Suppression bougies incompletes (deja fait)
- Controle prix negatifs
- Detection gaps anormaux
- Rapport qualite final
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class M15Cleaner:
    """Nettoyage et controle qualite des donnees M15"""
    
    def __init__(self, data_path: str = "data/processed"):
        self.data_path = Path(data_path)
    
    def load_m15(self, year: int) -> pd.DataFrame:
        """Charge les donnees M15 agregees"""
        file_path = self.data_path / f"m15_{year}.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier {file_path} introuvable")
        
        df = pd.read_csv(file_path)
        df['timestamp_15m'] = pd.to_datetime(df['timestamp_15m'])
        
        print(f"M15 {year} charge: {len(df):,} lignes")
        
        return df
    
    def check_negative_prices(self, df: pd.DataFrame, year: int) -> Dict:
        """Controle les prix negatifs ou nuls"""
        
        print(f"\nControle prix negatifs - {year}")
        print("-" * 60)
        
        issues = []
        
        for col in ['open_15m', 'high_15m', 'low_15m', 'close_15m']:
            neg_count = (df[col] <= 0).sum()
            if neg_count > 0:
                print(f"  ERREUR: {col} <= 0 sur {neg_count} lignes")
                issues.append({
                    'column': col,
                    'count': int(neg_count),
                    'indices': df[df[col] <= 0].index.tolist()[:10]
                })
            else:
                print(f"  {col}: OK")
        
        if not issues:
            print("  Resultat: AUCUN prix negatif")
        
        return {'year': year, 'negative_prices': issues}
    
    def check_gaps(self, df: pd.DataFrame, year: int) -> Dict:
        """Detecte les gaps anormaux"""
        
        print(f"\nDetection gaps anormaux - {year}")
        print("-" * 60)
        
        df_sorted = df.sort_values('timestamp_15m').reset_index(drop=True)
        time_diffs = df_sorted['timestamp_15m'].diff()
        
        # Categoriser les gaps
        expected = pd.Timedelta(minutes=15)
        small_gaps = time_diffs[(time_diffs > expected) & (time_diffs <= pd.Timedelta(hours=2))]
        medium_gaps = time_diffs[(time_diffs > pd.Timedelta(hours=2)) & (time_diffs <= pd.Timedelta(hours=24))]
        large_gaps = time_diffs[time_diffs > pd.Timedelta(hours=24)]
        
        print(f"  Gaps 15min-2h    : {len(small_gaps):,}")
        print(f"  Gaps 2h-24h      : {len(medium_gaps):,}")
        print(f"  Gaps > 24h       : {len(large_gaps):,} (week-ends normaux)")
        
        # Gaps anormaux (hors week-end)
        # Un gap est anormal si > 2h en semaine
        df_sorted['day_of_week'] = df_sorted['timestamp_15m'].dt.dayofweek
        
        abnormal_gaps = []
        for idx in medium_gaps.index:
            if idx > 0:
                day = df_sorted.loc[idx-1, 'day_of_week']
                # Lundi-Vendredi (0-4)
                if day < 5:
                    abnormal_gaps.append({
                        'index': int(idx),
                        'start': str(df_sorted.loc[idx-1, 'timestamp_15m']),
                        'end': str(df_sorted.loc[idx, 'timestamp_15m']),
                        'duration': str(time_diffs.loc[idx])
                    })
        
        if abnormal_gaps:
            print(f"\n  ATTENTION: {len(abnormal_gaps)} gaps anormaux en semaine")
            for gap in abnormal_gaps[:5]:
                print(f"    {gap['start']} -> {gap['end']} ({gap['duration']})")
        else:
            print(f"\n  Resultat: Aucun gap anormal detecte")
        
        return {
            'year': year,
            'small_gaps': int(len(small_gaps)),
            'medium_gaps': int(len(medium_gaps)),
            'large_gaps': int(len(large_gaps)),
            'abnormal_gaps': abnormal_gaps
        }
    
    def check_ohlc_coherence(self, df: pd.DataFrame, year: int) -> Dict:
        """Verifie la coherence OHLC"""
        
        print(f"\nCoherence OHLC - {year}")
        print("-" * 60)
        
        issues = []
        
        # High < Low
        high_low = (df['high_15m'] < df['low_15m']).sum()
        if high_low > 0:
            print(f"  ERREUR: High < Low sur {high_low} lignes")
            issues.append('high_low')
        else:
            print(f"  High >= Low: OK")
        
        # Close hors [Low, High]
        close_out = ((df['close_15m'] < df['low_15m']) | (df['close_15m'] > df['high_15m'])).sum()
        if close_out > 0:
            print(f"  ERREUR: Close hors [Low,High] sur {close_out} lignes")
            issues.append('close_out_of_bounds')
        else:
            print(f"  Close dans [Low,High]: OK")
        
        # Open hors [Low, High]
        open_out = ((df['open_15m'] < df['low_15m']) | (df['open_15m'] > df['high_15m'])).sum()
        if open_out > 0:
            print(f"  ERREUR: Open hors [Low,High] sur {open_out} lignes")
            issues.append('open_out_of_bounds')
        else:
            print(f"  Open dans [Low,High]: OK")
        
        if not issues:
            print(f"\n  Resultat: Coherence OHLC parfaite")
        
        return {'year': year, 'ohlc_issues': issues}
    
    def generate_quality_report(self, year: int) -> Dict:
        """Genere un rapport qualite complet"""
        
        print(f"\n{'='*60}")
        print(f"RAPPORT QUALITE - {year}")
        print(f"{'='*60}")
        
        df = self.load_m15(year)
        
        # Controles
        negative_prices = self.check_negative_prices(df, year)
        gaps = self.check_gaps(df, year)
        ohlc = self.check_ohlc_coherence(df, year)
        
        # Statistiques
        print(f"\nStatistiques descriptives - {year}")
        print("-" * 60)
        print(f"  Nombre de bougies: {len(df):,}")
        print(f"  Periode: {df['timestamp_15m'].min()} -> {df['timestamp_15m'].max()}")
        print(f"  Duree: {(df['timestamp_15m'].max() - df['timestamp_15m'].min()).days} jours")
        print(f"  Prix moyen: {df['close_15m'].mean():.5f}")
        print(f"  Volatilite: {df['close_15m'].std():.5f}")
        print(f"  Min: {df['close_15m'].min():.5f}")
        print(f"  Max: {df['close_15m'].max():.5f}")
        
        if 'n_candles_m1' in df.columns:
            print(f"\n  Completude (bougies M1 par M15):")
            print(f"    Moyenne: {df['n_candles_m1'].mean():.2f}")
            print(f"    Min: {df['n_candles_m1'].min()}")
            print(f"    Max: {df['n_candles_m1'].max()}")
        
        # Verdict final
        print(f"\n{'='*60}")
        all_clear = (
            not negative_prices['negative_prices'] and
            not ohlc['ohlc_issues'] and
            not gaps['abnormal_gaps']
        )
        
        if all_clear:
            print("VERDICT: DONNEES PROPRES - PRET POUR FEATURE ENGINEERING")
        else:
            print("VERDICT: PROBLEMES DETECTES - NETTOYAGE REQUIS")
        print(f"{'='*60}\n")
        
        # Rapport complet
        report = {
            'year': year,
            'generated_at': datetime.now().isoformat(),
            'n_candles': len(df),
            'period': {
                'start': str(df['timestamp_15m'].min()),
                'end': str(df['timestamp_15m'].max()),
                'days': (df['timestamp_15m'].max() - df['timestamp_15m'].min()).days
            },
            'statistics': {
                'close_mean': float(df['close_15m'].mean()),
                'close_std': float(df['close_15m'].std()),
                'close_min': float(df['close_15m'].min()),
                'close_max': float(df['close_15m'].max())
            },
            'quality_checks': {
                'negative_prices': negative_prices,
                'gaps': gaps,
                'ohlc_coherence': ohlc
            },
            'verdict': 'CLEAN' if all_clear else 'ISSUES_DETECTED'
        }
        
        return report
    
    def process_all_years(self) -> Dict[int, Dict]:
        """Genere les rapports pour toutes les annees"""
        
        reports = {}
        
        for year in [2022, 2023, 2024]:
            try:
                report = self.generate_quality_report(year)
                reports[year] = report
            except Exception as e:
                print(f"\nERREUR {year}: {e}")
        
        return reports
    
    def save_reports(self, reports: Dict[int, Dict]):
        """Sauvegarde les rapports en JSON"""
        
        output_file = self.data_path / "quality_report_m15.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(reports, f, indent=2, ensure_ascii=False)
        
        print(f"Rapports sauvegardes: {output_file}")


def main():
    """Fonction principale"""
    
    print("="*60)
    print("PHASE 3 - NETTOYAGE M15")
    print("="*60)
    
    cleaner = M15Cleaner()
    reports = cleaner.process_all_years()
    cleaner.save_reports(reports)
    
    print(f"\n{'='*60}")
    print("RESUME GLOBAL")
    print(f"{'='*60}\n")
    
    for year, report in reports.items():
        print(f"{year}: {report['verdict']}")
    
    print("\nRapport JSON genere dans: data/processed/quality_report_m15.json")
    
    return reports


if __name__ == "__main__":
    reports = main()