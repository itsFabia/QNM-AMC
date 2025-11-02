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
print(f'  IC Mean       : {new["IC_mean"].mean():.4f}')
print(f'  IC Median     : {new["IC_mean"].median():.4f}')
print(f'  IC Std        : {new["IC_mean"].std():.4f}')
print(f'  Hit-Rate@K    : {new["HitRate_topk"].mean():.4f} ({new["HitRate_topk"].mean()*100:.2f}%)')
print(f'  Sharpe Mean   : {new["Sharpe"].mean():.2f}')
print(f'  CAGR Mean     : {new["CAGR"].mean():.2f}%')
print(f'  Ø Days/Split  : {new["n_days"].mean():.1f}')

if has_old:
    print('\n' + '='*60)
    print('ALTE METRIKEN (VOR LEAKAGE-FIXES)')
    print('='*60)
    print(f'  IC Mean       : {old["IC_mean"].mean():.4f}')
    print(f'  IC Median     : {old["IC_mean"].median():.4f}')
    print(f'  Hit-Rate@K    : {old["HitRate_topk"].mean():.4f} ({old["HitRate_topk"].mean()*100:.2f}%)')
    print(f'  Sharpe Mean   : {old["Sharpe"].mean():.2f}')
    print(f'  CAGR Mean     : {old["CAGR"].mean():.2f}%')

    print('\n' + '='*60)
    print('UNTERSCHIED (NEU - ALT)')
    print('='*60)
    diff_ic = new["IC_mean"].mean() - old["IC_mean"].mean()
    diff_hit = (new["HitRate_topk"].mean() - old["HitRate_topk"].mean()) * 100
    diff_sharpe = new["Sharpe"].mean() - old["Sharpe"].mean()
    diff_cagr = new["CAGR"].mean() - old["CAGR"].mean()
    print(f'  IC Mean       : {diff_ic:+.4f} ({diff_ic/old["IC_mean"].mean()*100:+.1f}%)')
    print(f'  Hit-Rate@K    : {diff_hit:+.2f}%-Punkte')
    print(f'  Sharpe        : {diff_sharpe:+.2f}')
    print(f'  CAGR          : {diff_cagr:+.2f}%-Punkte')

print('\n' + '='*60)
print('SPLIT DETAILS (Best/Worst)')
print('='*60)
best_idx = new["IC_mean"].idxmax()
worst_idx = new["IC_mean"].idxmin()
print(f'Best Split  : #{new.loc[best_idx, "split_id"]} -> IC={new.loc[best_idx, "IC_mean"]:.4f}, HitRate={new.loc[best_idx, "HitRate_topk"]:.4f}')
print(f'Worst Split : #{new.loc[worst_idx, "split_id"]} -> IC={new.loc[worst_idx, "IC_mean"]:.4f}, HitRate={new.loc[worst_idx, "HitRate_topk"]:.4f}')

print('\n' + '='*60)
