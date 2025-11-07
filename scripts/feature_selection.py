"""
Feature Selection Script - Identifiziert Top Features via LightGBM Feature Importance
Läuft auf den ersten 10 Splits, aggregiert Feature Importance, selektiert Top 30
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

def train_and_get_importance(train_parquet, target_col='Excess_5d_fwd'):
    """
    Trainiert ein LightGBM Modell und gibt Feature Importance zurück
    """
    df = pd.read_parquet(train_parquet)

    # Prepare features and target
    exclude_cols = ['Date', 'Ticker', 'Ret_5d_fwd', 'Bench_Ret_5d_fwd',
                    'Excess_5d_fwd', 'Excess_ret', 'Return_raw', 'Bench_Return_raw']

    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Remove rows where TARGET is NaN, fill feature NaNs with 0 (simple imputation for feature importance)
    df_clean = df[df[target_col].notna()].copy()

    # Fill feature NaNs
    df_clean[feature_cols] = df_clean[feature_cols].fillna(0)

    X = df_clean[feature_cols].values
    y = df_clean[target_col].values

    print(f"  Training shape: {X.shape}, Features: {len(feature_cols)}")

    # Train LightGBM Regressor (not Ranker, since we have continuous targets)
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'min_data_in_leaf': 50,
        'verbose': -1
    }

    train_data = lgb.Dataset(X, label=y)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data],
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(period=0)]
    )

    # Get feature importance
    importance = model.feature_importance(importance_type='gain')

    return dict(zip(feature_cols, importance))


def aggregate_importance_across_splits(parquet_dir, n_splits=10, target_col='Excess_5d_fwd'):
    """
    Lädt erste n_splits, trainiert Modelle, aggregiert Feature Importance
    """
    parquet_dir = Path(parquet_dir)

    # Find all train parquet files (try both naming patterns)
    train_files = sorted(parquet_dir.glob('split_*_train.parquet'))[:n_splits]

    if not train_files:
        train_files = sorted(parquet_dir.glob('scaled_train_*.parquet'))[:n_splits]

    if not train_files:
        raise ValueError(f"No train parquet files found in {parquet_dir}")

    print(f"[INFO] Found {len(train_files)} train splits")
    print(f"[INFO] Using first {n_splits} splits for feature selection")

    all_importance = []

    for i, train_file in enumerate(train_files):
        print(f"[INFO] Processing split {i}: {train_file.name}")
        try:
            importance_dict = train_and_get_importance(train_file, target_col)
            all_importance.append(importance_dict)
        except Exception as e:
            print(f"  [WARN] Failed: {e}")
            continue

    if not all_importance:
        raise ValueError("No successful importance calculations")

    # Aggregate importance (mean across splits)
    all_features = set()
    for imp_dict in all_importance:
        all_features.update(imp_dict.keys())

    aggregated_importance = {}
    for feature in all_features:
        importance_values = [imp_dict.get(feature, 0.0) for imp_dict in all_importance]
        aggregated_importance[feature] = np.mean(importance_values)

    # Sort by importance
    sorted_features = sorted(aggregated_importance.items(), key=lambda x: x[1], reverse=True)

    return sorted_features


def select_top_features_and_filter_data(input_csv, output_csv, top_features,
                                        keep_cols=None):
    """
    Filtert Dataset auf Top Features
    """
    print(f"[INFO] Loading {input_csv}")

    try:
        df = pd.read_csv(input_csv, sep=";")
        if df.shape[1] == 1:
            df = pd.read_csv(input_csv)
    except:
        df = pd.read_csv(input_csv)

    print(f"[INFO] Original shape: {df.shape}")

    # Columns to always keep
    if keep_cols is None:
        keep_cols = ['Date', 'Ticker', 'Price', 'Volume', 'Ret_5d_fwd',
                     'Bench_Ret_5d_fwd', 'Excess_5d_fwd', 'Excess_ret',
                     'Return_raw', 'Bench_Return_raw']

    # Ensure all keep_cols exist
    keep_cols = [c for c in keep_cols if c in df.columns]

    # Select top features that exist in the dataframe
    selected_features = [f for f in top_features if f in df.columns]

    # Combine
    final_cols = list(set(keep_cols + selected_features))

    df_filtered = df[final_cols].copy()

    print(f"[INFO] Filtered shape: {df_filtered.shape}")
    print(f"[INFO] Kept {len(selected_features)} selected features + {len(keep_cols)} base columns")

    # Backup original
    backup_path = Path(output_csv).parent / (Path(output_csv).stem + "_BEFORE_SELECTION.csv")
    if not backup_path.exists():
        df.to_csv(backup_path, index=False)
        print(f"[INFO] Backup created: {backup_path}")

    # Save filtered
    df_filtered.to_csv(output_csv, index=False)
    print(f"[INFO] Saved filtered dataset: {output_csv}")

    return df_filtered


def main():
    parser = argparse.ArgumentParser(description='Feature Selection via LightGBM Importance')
    parser.add_argument('--parquet-dir', type=str, default='reports/scaled',
                        help='Directory with scaled parquet files')
    parser.add_argument('--input-csv', type=str, default='data/AMC_model_input_reduced.csv',
                        help='Input CSV file with all features')
    parser.add_argument('--output-csv', type=str, default='data/AMC_model_input_selected.csv',
                        help='Output CSV file with selected features')
    parser.add_argument('--top-k', type=int, default=30,
                        help='Number of top features to select')
    parser.add_argument('--n-splits', type=int, default=10,
                        help='Number of splits to use for importance calculation')
    parser.add_argument('--target', type=str, default='Excess_5d_fwd',
                        help='Target column')

    args = parser.parse_args()

    print("="*70)
    print("FEATURE SELECTION VIA LIGHTGBM IMPORTANCE")
    print("="*70)

    # Step 1: Calculate feature importance
    print(f"\n[STEP 1] Calculating feature importance on {args.n_splits} splits")
    sorted_features = aggregate_importance_across_splits(
        args.parquet_dir,
        n_splits=args.n_splits,
        target_col=args.target
    )

    # Step 2: Select top K
    print(f"\n[STEP 2] Selecting top {args.top_k} features")
    top_features = [f[0] for f in sorted_features[:args.top_k]]

    print(f"\nTop {args.top_k} Features (by importance):")
    for i, (feature, importance) in enumerate(sorted_features[:args.top_k], 1):
        print(f"  {i:2d}. {feature:40s} | Importance: {importance:10.2f}")

    # Save importance ranking
    importance_df = pd.DataFrame(sorted_features, columns=['Feature', 'Importance'])
    importance_path = Path(args.output_csv).parent / 'feature_importance_ranking.csv'
    importance_df.to_csv(importance_path, index=False)
    print(f"\n[INFO] Full importance ranking saved: {importance_path}")

    # Step 3: Filter dataset
    print(f"\n[STEP 3] Filtering dataset to top {args.top_k} features")
    df_filtered = select_top_features_and_filter_data(
        args.input_csv,
        args.output_csv,
        top_features
    )

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Original features: {len(sorted_features)}")
    print(f"Selected features: {args.top_k}")
    print(f"Final dataset: {df_filtered.shape}")
    print(f"Output: {args.output_csv}")
    print("\n[DONE] Feature selection complete!")

    print(f"\nNext step:")
    print(f"  python scripts/scale_and_save.py --csv {args.output_csv} \\")
    print(f"    --splits-log reports/splits_log.csv --outdir reports/scaled_selected \\")
    print(f"    --target {args.target} --winsorize --winsorize-sigma 3.0 \\")
    print(f"    --winsorize-patterns \"__logdiff1$,__chgstd20$\" \\")
    print(f"    --reduce-features --corr-threshold 0.95")


if __name__ == "__main__":
    main()
