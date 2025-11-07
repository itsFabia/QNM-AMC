"""
Bootstrap Confidence Intervals für Sharpe Ratio
Ziel: Verstehen der Unsicherheit in der Performance-Metrik
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json

def load_split_results(eval_dir):
    """
    Lädt alle Split-Ergebnisse aus eval_details.csv
    """
    eval_dir = Path(eval_dir)
    details_file = eval_dir / 'eval_details.csv'

    if not details_file.exists():
        raise FileNotFoundError(f"eval_details.csv not found in {eval_dir}")

    df = pd.read_csv(details_file)
    print(f"[INFO] Loaded {len(df)} splits from {details_file}")

    return df


def calculate_sharpe(returns):
    """
    Berechnet Sharpe Ratio aus Returns
    """
    if len(returns) == 0:
        return np.nan

    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)

    if std_ret == 0:
        return np.nan

    # Annualisiert (12 Monate pro Jahr, da monatliche Splits)
    sharpe = (mean_ret / std_ret) * np.sqrt(12)

    return sharpe


def bootstrap_sharpe_ci(df, n_bootstrap=10000, confidence_level=0.95):
    """
    Bootstrap Confidence Intervals für Sharpe Ratio
    """
    print(f"\n[BOOTSTRAP] Running {n_bootstrap} bootstrap iterations...")

    # Extrahiere CAGR pro Split (als Proxy für Returns)
    # Falls CAGR nicht verfügbar, nutze andere Metriken

    if 'CAGR' in df.columns:
        returns = df['CAGR'].values
    elif 'Sharpe' in df.columns:
        # Fallback: reconstruct returns from Sharpe
        print("[WARN] CAGR not found, using Sharpe directly")
        sharpes = df['Sharpe'].values
        returns = sharpes  # Direct Sharpe values
    else:
        raise ValueError("Neither CAGR nor Sharpe found in eval_details.csv")

    # Remove NaNs
    returns = returns[~np.isnan(returns)]

    print(f"[INFO] Using {len(returns)} valid returns")
    print(f"[INFO] Mean Return: {np.mean(returns):.4f}")
    print(f"[INFO] Std Return: {np.std(returns, ddof=1):.4f}")

    # Original Sharpe
    original_sharpe = calculate_sharpe(returns)
    print(f"\n[ORIGINAL] Sharpe Ratio: {original_sharpe:.4f}")

    # Bootstrap
    bootstrap_sharpes = []

    for i in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = np.random.choice(returns, size=len(returns), replace=True)
        bootstrap_sharpe = calculate_sharpe(bootstrap_sample)

        if not np.isnan(bootstrap_sharpe):
            bootstrap_sharpes.append(bootstrap_sharpe)

        if (i + 1) % 1000 == 0:
            print(f"  Progress: {i+1}/{n_bootstrap}")

    bootstrap_sharpes = np.array(bootstrap_sharpes)

    # Calculate confidence intervals
    alpha = (1 - confidence_level) / 2
    lower_percentile = alpha * 100
    upper_percentile = (1 - alpha) * 100

    ci_lower = np.percentile(bootstrap_sharpes, lower_percentile)
    ci_upper = np.percentile(bootstrap_sharpes, upper_percentile)

    # Statistics
    bootstrap_mean = np.mean(bootstrap_sharpes)
    bootstrap_std = np.std(bootstrap_sharpes)

    print(f"\n{'='*70}")
    print("BOOTSTRAP RESULTS")
    print(f"{'='*70}")
    print(f"Original Sharpe:        {original_sharpe:.4f}")
    print(f"Bootstrap Mean Sharpe:  {bootstrap_mean:.4f}")
    print(f"Bootstrap Std:          {bootstrap_std:.4f}")
    print(f"\n{confidence_level*100:.0f}% Confidence Interval:")
    print(f"  Lower Bound:  {ci_lower:.4f}")
    print(f"  Upper Bound:  {ci_upper:.4f}")
    print(f"  Range:        {ci_upper - ci_lower:.4f}")

    # Probability of positive Sharpe
    prob_positive = np.mean(bootstrap_sharpes > 0) * 100
    print(f"\nProbability of Positive Sharpe: {prob_positive:.1f}%")

    # Percentiles
    print(f"\nPercentiles:")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        val = np.percentile(bootstrap_sharpes, p)
        print(f"  {p:2d}th: {val:7.4f}")

    return {
        'original_sharpe': original_sharpe,
        'bootstrap_mean': bootstrap_mean,
        'bootstrap_std': bootstrap_std,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_range': ci_upper - ci_lower,
        'prob_positive': prob_positive,
        'n_bootstrap': n_bootstrap,
        'n_splits': len(returns),
        'confidence_level': confidence_level
    }


def main():
    parser = argparse.ArgumentParser(description='Bootstrap Confidence Intervals for Sharpe Ratio')
    parser.add_argument('--eval-dir', type=str, required=True,
                        help='Directory with eval_details.csv')
    parser.add_argument('--n-bootstrap', type=int, default=10000,
                        help='Number of bootstrap iterations')
    parser.add_argument('--confidence', type=float, default=0.95,
                        help='Confidence level (e.g., 0.95 for 95%)')

    args = parser.parse_args()

    print("="*70)
    print("BOOTSTRAP CONFIDENCE INTERVALS - SHARPE RATIO")
    print("="*70)

    # Load data
    df = load_split_results(args.eval_dir)

    # Bootstrap
    results = bootstrap_sharpe_ci(
        df,
        n_bootstrap=args.n_bootstrap,
        confidence_level=args.confidence
    )

    # Save results
    output_dir = Path(args.eval_dir)
    output_file = output_dir / 'bootstrap_sharpe_ci.json'

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[SAVED] Results saved to: {output_file}")

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    ci_range = results['ci_range']
    if ci_range < 0.3:
        print("✅ TIGHT CI: Performance is relatively stable")
    elif ci_range < 0.6:
        print("⚠️  MODERATE CI: Some uncertainty in performance")
    else:
        print("❌ WIDE CI: High uncertainty - be cautious!")

    if results['prob_positive'] > 80:
        print(f"✅ HIGH CONFIDENCE: {results['prob_positive']:.1f}% chance of positive Sharpe")
    elif results['prob_positive'] > 60:
        print(f"⚠️  MODERATE CONFIDENCE: {results['prob_positive']:.1f}% chance of positive Sharpe")
    else:
        print(f"❌ LOW CONFIDENCE: Only {results['prob_positive']:.1f}% chance of positive Sharpe")


if __name__ == "__main__":
    main()
