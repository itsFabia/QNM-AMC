import pandas as pd
import numpy as np

# Neue Metriken laden
new = pd.read_csv('reports/eval/eval_summary.csv')

# Alte Metriken laden (falls vorhanden)
try:
    old = pd.read_csv('reports/eval_WITH_LEAKAGE_BUG/eval_summary.csv')
    has_old = True
except:
    has_old = False

print('='*60)
print('NEUE METRIKEN (MIT LEAKAGE-FIXES)')
print('='*60)
print(f'Anzahl Splits: {len(new)}')
print(f'\nRANKER (LambdaRank):')
print(f'  IC Mean       : {new["rank_IC_mean"].mean():.4f}')
print(f'  IC Median     : {new["rank_IC_median"].median():.4f}')
print(f'  IC Std        : {new["rank_IC_mean"].std():.4f}')
print(f'  Hit-Rate@K    : {new["rank_HitRate@K"].mean():.4f} ({new["rank_HitRate@K"].mean()*100:.2f}%)')
print(f'  Ø Days/Split  : {new["rank_days"].mean():.1f}')

print(f'\nREGRESSOR:')
print(f'  IC Mean       : {new["reg_IC_mean"].mean():.4f}')
print(f'  IC Median     : {new["reg_IC_median"].median():.4f}')
print(f'  Hit-Rate@K    : {new["reg_HitRate@K"].mean():.4f} ({new["reg_HitRate@K"].mean()*100:.2f}%)')

if has_old:
    print('\n' + '='*60)
    print('ALTE METRIKEN (VOR LEAKAGE-FIXES)')
    print('='*60)
    print(f'  IC Mean       : {old["rank_IC_mean"].mean():.4f}')
    print(f'  IC Median     : {old["rank_IC_median"].median():.4f}')
    print(f'  Hit-Rate@K    : {old["rank_HitRate@K"].mean():.4f} ({old["rank_HitRate@K"].mean()*100:.2f}%)')

    print('\n' + '='*60)
    print('UNTERSCHIED (NEU - ALT)')
    print('='*60)
    diff_ic = new["rank_IC_mean"].mean() - old["rank_IC_mean"].mean()
    diff_hit = (new["rank_HitRate@K"].mean() - old["rank_HitRate@K"].mean()) * 100
    print(f'  IC Mean       : {diff_ic:+.4f} ({diff_ic/old["rank_IC_mean"].mean()*100:+.1f}%)')
    print(f'  Hit-Rate@K    : {diff_hit:+.2f}%')

print('\n' + '='*60)
print('SPLIT DETAILS (Best/Worst)')
print('='*60)
best_idx = new["rank_IC_mean"].idxmax()
worst_idx = new["rank_IC_mean"].idxmin()
print(f'Best Split  : #{best_idx} -> IC={new.loc[best_idx, "rank_IC_mean"]:.4f}, HitRate={new.loc[best_idx, "rank_HitRate@K"]:.4f}')
print(f'Worst Split : #{worst_idx} -> IC={new.loc[worst_idx, "rank_IC_mean"]:.4f}, HitRate={new.loc[worst_idx, "rank_HitRate@K"]:.4f}')

print('\n' + '='*60)
