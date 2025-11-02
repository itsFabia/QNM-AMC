import json

# Lade Metadata vom ersten Split
with open('reports/scaled/meta_000.json') as f:
    meta = json.load(f)

print('='*60)
print('FEATURES IM SCALED DATASET')
print('='*60)
print(f'Anzahl Features: {len(meta["active_features"])}')
print()
print('Erste 30 Features:')
for i, feat in enumerate(meta['active_features'][:30], 1):
    print(f'  {i:2d}. {feat}')

# Prüfe ob Ret_5d_fwd oder Bench_Ret_5d_fwd dabei ist
print()
print('='*60)
print('LEAKAGE CHECK')
print('='*60)
if 'Ret_5d_fwd' in meta['active_features']:
    print('⚠️  WARNING: Ret_5d_fwd ist ein Feature (LEAKAGE!)')
else:
    print('✅ Ret_5d_fwd ist NICHT im Feature-Set')

if 'Bench_Ret_5d_fwd' in meta['active_features']:
    print('⚠️  WARNING: Bench_Ret_5d_fwd ist ein Feature (LEAKAGE!)')
else:
    print('✅ Bench_Ret_5d_fwd ist NICHT im Feature-Set')

print()
print(f'Target: {meta["target"]}')
print()
print('='*60)
print('PREPROCESSING INFO')
print('='*60)
print(f'Winsorizing aktiviert: {meta.get("winsorize_enabled", False)}')
print(f'Feature Reduction aktiviert: {meta.get("reduce_features_enabled", False)}')
if meta.get("dropped_features"):
    print(f'Anzahl gedropter Features: {len(meta["dropped_features"])}')
    print(f'Erste 10 gedropte Features:')
    for feat in meta["dropped_features"][:10]:
        print(f'  - {feat}')
