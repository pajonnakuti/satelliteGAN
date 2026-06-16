#!/usr/bin/env python3
"""
59_model_comparison.py  —  Multi-Model Comparison: LSTM | N-BEATS | MOIRAI
════════════════════════════════════════════════════════════════════════════
Reads predictions from each model's zip archive or extracted output directory
and produces:

  Plot A: Taylor Diagram        — R, normalised σ, centred RMSE (Overall only)
  Plot B: Error Density Curves  — KDE of daily errors (range 0-1, extreme left RMSE / extreme right R)

Usage: run after all three forecast scripts have completed.
Output → comparison-outputs/
"""

import os
import sys
import zipfile
import io
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import warnings; warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# 0. CONFIG
# ══════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = Path(__file__).resolve().parent
BASE = str(SCRIPT_DIR.parent)
OUT = SCRIPT_DIR / "comparison-outputs"
OUT.mkdir(parents=True, exist_ok=True)
ROLLING_CSVS = {
    'LSTM':  SCRIPT_DIR / 'rolling_predictions_56.csv',
    'N-BEATS': SCRIPT_DIR / 'rolling_predictions_57.csv',
    'MOIRAI': SCRIPT_DIR / 'rolling_predictions_58f.csv',
}

# Google Colab auto-setup/extraction
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    print("Detected Google Colab environment. Checking for output zip archives...")
    import zipfile
    
    colab_zips = {
        '56_lstm_rolling_7day (8).zip': '/content/outputs/56_lstm_rolling_7day',
        '57_nbeats_rolling_7day (6).zip': '/content/outputs/57_nbeats_rolling_7day',
        '58f_moirai_regional_gradient_outputs (2).zip': '/content/outputs/58f_moirai_regional_gradient',
    }
    
    for zname, dest_dir in colab_zips.items():
        paths_to_check = [
            Path(f'/content/{zname}'),
            Path(f'/content/drive/MyDrive/{zname}'),
            Path(BASE) / zname
        ]
        
        extracted = False
        for p in paths_to_check:
            if p.exists():
                print(f"  Found {zname} at {p}. Extracting to {dest_dir}...")
                os.makedirs(dest_dir, exist_ok=True)
                with zipfile.ZipFile(p, 'r') as zf:
                    zf.extractall(dest_dir)
                extracted = True
                break
        if not extracted:
            print(f"  Note: {zname} not found, skipping extraction.")

MODELS_CONFIG = {
    'LSTM': {
        'prefixes': ['56_lstm'],
        'dirs': ['outputs/lstm-outputs', 'outputs/56_lstm_rolling_7day', 'outputs/56_lstm_rolling_7day_outputs'],
        'color': '#2980B9',
        'marker': 'o',
        'ls': '-'
    },
    'N-BEATS': {
        'prefixes': ['57_nbeats'],
        'dirs': ['outputs/nbeats-outputs', 'outputs/57_nbeats_rolling_7day', 'outputs/57_nbeats_rolling_7day_outputs'],
        'color': '#27AE60',
        'marker': 's',
        'ls': '--'
    },
    'MOIRAI': {
        'prefixes': ['58f_moirai'],
        'dirs': ['outputs/moirai-outputs', 'outputs/58f_moirai_regional_gradient', '58f_outputs', '58e_outputs_tuned', 'outputs/58g_moirai_detrended_residuals'],
        'color': '#E74C3C',
        'marker': '^',
        'ls': '-.'
    }
}

print("=== 59_model_comparison.py ===")

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA (Robust searching for Zips or Directories)
# ══════════════════════════════════════════════════════════════════════════════
dfs = {}

def load_data(name, cfg):
    # Try pre-saved rolling_predictions CSV in model_comparison/ first
    csv_path = ROLLING_CSVS.get(name)
    if csv_path and csv_path.exists():
        print(f"  {name}: Found pre-saved predictions -> {csv_path}")
        try:
            df = pd.read_csv(csv_path, parse_dates=['date'])
            return df
        except Exception as e:
            print(f"    Error reading {csv_path}: {e}")

    for prefix in cfg['prefixes']:
        zips = list(Path(BASE).glob(f"{prefix}*.zip"))
        
        if zips:
            def sort_key(p):
                import re
                m = re.search(r'\((\d+)\)', p.name)
                val = int(m.group(1)) if m else 0
                return (val, p.name)
            zips.sort(key=sort_key, reverse=True)
            zf_path = zips[0]
            print(f"  {name}: Found matching zip file -> {zf_path.name}")
            try:
                with zipfile.ZipFile(zf_path) as zf:
                    csv_files = [f for f in zf.namelist() if f.endswith('.csv') and ('prediction' in f or ('summary' not in f and 'loss' not in f and 'bias' not in f))]
                    if csv_files:
                        pred_csv = [f for f in csv_files if 'prediction' in f][0] if any('prediction' in f for f in csv_files) else csv_files[0]
                        df = pd.read_csv(io.BytesIO(zf.read(pred_csv)), parse_dates=['date'])
                        print(f"    Loaded {pred_csv} from zip.")
                        return df
            except Exception as e:
                print(f"    Error reading zip {zf_path.name}: {e}")

    for folder in cfg['dirs']:
        fp_dir = Path(BASE) / folder
        for csv_name in ['daily_predictions.csv', 'rolling_predictions.csv']:
            fp = fp_dir / csv_name
            if fp.exists():
                print(f"  {name}: Found directory data -> {fp}")
                try:
                    df = pd.read_csv(fp, parse_dates=['date'])
                    return df
                except Exception as e:
                    print(f"    Error reading CSV {fp}: {e}")
                    
    return None

for name, cfg in MODELS_CONFIG.items():
    df = load_data(name, cfg)
    if df is not None:
        if 'ground_truth' not in df.columns:
            if 'actual' in df.columns:
                df = df.rename(columns={'actual': 'ground_truth'})
            elif 'actual_sst' in df.columns:
                df = df.rename(columns={'actual_sst': 'ground_truth'})
                
        if 'predicted_avg' not in df.columns:
            if 'predicted' in df.columns:
                df = df.rename(columns={'predicted': 'predicted_avg'})
            else:
                day_cols = [c for c in df.columns if c.startswith('pred_day')]
                if day_cols:
                    df['predicted_avg'] = df[day_cols].mean(axis=1)

        df['error'] = df['predicted_avg'] - df['ground_truth']
        
        if 'month' not in df.columns and 'date' in df.columns:
            df['month'] = df['date'].dt.month
            
        dfs[name] = df
        rmse = np.sqrt((df['error']**2).mean())
        print(f"    Total rows: {len(df)}  RMSE: {rmse:.4f}°C")
    else:
        print(f"  WARNING: Could not locate data for model {name}")

if len(dfs) < 2:
    print("Need at least 2 model datasets to compare. Exiting.")
    sys.exit(1)

common_dates = None
for df in dfs.values():
    s = set(df['date'].dt.strftime('%Y-%m-%d').tolist())
    common_dates = s if common_dates is None else common_dates & s
common_dates = sorted(common_dates)
print(f"\nAligned on common prediction days: {len(common_dates)}")

for name in list(dfs.keys()):
    dfs[name] = dfs[name][dfs[name]['date'].dt.strftime('%Y-%m-%d').isin(common_dates)].reset_index(drop=True)

gt = dfs[list(dfs.keys())[0]]['ground_truth'].values
dates_arr = dfs[list(dfs.keys())[0]]['date'].values

# ══════════════════════════════════════════════════════════════════════════════
# 2. PRINT METRICS (Overall Only)
# ══════════════════════════════════════════════════════════════════════════════
metric_rows = []
for name, df in dfs.items():
    pred = df['predicted_avg'].values
    err  = df['error'].values
    r,_  = stats.pearsonr(gt, pred) if len(gt) > 2 else (np.nan, np.nan)
    metric_rows.append({
        'Model': name,
        'RMSE':  float(np.sqrt(np.mean(err**2))),
        'MAE':   float(np.mean(np.abs(err))),
        'Bias':  float(np.mean(err)),
        'R':     float(r),
        'R2':    float(r**2) if not np.isnan(r) else np.nan,
        'MaxErr':float(np.max(np.abs(err)))
    })

metric_df = pd.DataFrame(metric_rows)
print("\nSkill scores (Overall):")
print(metric_df[['Model','RMSE','MAE','Bias','R','R2','MaxErr']].to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 3. TAYLOR DIAGRAM (Plot A) — Overall Only
# ══════════════════════════════════════════════════════════════════════════════
print("\nPlot A: Taylor Diagram (Overall Only)")

def taylor_diagram(ax, models_data, title="Taylor Diagram"):
    for crmse in [0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0]:
        theta_c = np.linspace(0, np.pi, 300)
        xc = 1 + crmse*np.cos(theta_c)
        yc = crmse*np.sin(theta_c)
        valid = (xc**2+yc**2)**0.5 <= 2.0
        ax.plot(xc[valid], yc[valid], 'k-', lw=0.5, alpha=0.25)
        xlab = 1 + crmse*np.cos(np.pi*0.55)
        ylab = crmse*np.sin(np.pi*0.55)
        if 0 < xlab < 2.2 and 0 < ylab < 2.2:
            ax.text(xlab, ylab, f'{crmse:.1f}', fontsize=7, color='#555',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.6))

    for r_val in [0.5, 1.0, 1.5, 2.0]:
        theta_c = np.linspace(0, np.pi, 300)
        ax.plot(r_val*np.cos(theta_c), r_val*np.sin(theta_c),
                'b--' if r_val==1.0 else 'k:', lw=1.0 if r_val==1.0 else 0.5, alpha=0.35)
        ax.text(r_val*np.cos(np.radians(5)), r_val*np.sin(np.radians(5)),
                f'σ={r_val:.1f}', fontsize=7, color='#336', alpha=0.7)

    for r_corr in [0.0,0.2,0.4,0.6,0.7,0.8,0.9,0.95,0.99,1.0]:
        theta = np.arccos(r_corr)
        ax.plot([0, 2.1*np.cos(theta)], [0, 2.1*np.sin(theta)],
                'g-', lw=0.6, alpha=0.25)
        ax.text(1.75*np.cos(theta), 1.75*np.sin(theta)+0.04,
                f'R={r_corr}', fontsize=6.5, color='#282',
                ha='center', rotation=np.degrees(-theta)+90, alpha=0.8)

    ax.plot(1, 0, 'k*', ms=15, zorder=10, label='Observation (reference)')
    ax.text(1.0, -0.09, 'OBS', ha='center', fontsize=9, fontweight='bold', color='k')

    for mname, (R, std_norm, crmse_norm) in models_data.items():
        theta  = np.arccos(np.clip(R, -1, 1))
        x_pos  = std_norm * np.cos(theta)
        y_pos  = std_norm * np.sin(theta)
        cfg_m  = MODELS_CONFIG.get(mname, {})
        col    = cfg_m.get('color', 'gray')
        mrk    = cfg_m.get('marker', 'o')
        ax.plot(x_pos, y_pos, mrk, ms=14, color=col, markeredgecolor='black',
                markeredgewidth=0.8, zorder=8,
                label=f'{mname}  R={R:.3f}  σ={std_norm:.3f}  CRMSE={crmse_norm:.3f}')
        ax.annotate(mname, (x_pos, y_pos), fontsize=9, fontweight='bold',
                    xytext=(x_pos+0.05, y_pos+0.04), color=col)

    ax.set_xlim(-0.1, 2.1); ax.set_ylim(-0.05, 2.1)
    ax.set_xlabel('Normalised Standard Deviation (σ_pred / σ_obs)', fontsize=11)
    ax.set_ylabel('Normalised Standard Deviation (σ_pred / σ_obs)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.90)
    ax.set_aspect('equal')
    ax.grid(which='major', alpha=0.12, ls='--')
    ax.text(0.02, 0.97, 'Dashed lines: normalised σ\nGreen lines: R contours\n'
             'Black arcs: CRMSE contours\n(centred on OBS point)',
             transform=ax.transAxes, fontsize=7.5, va='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.80))

sigma_obs = float(np.std(gt))
taylor_data_overall = {}
for name, df in dfs.items():
    pred = df['predicted_avg'].values
    R, _ = stats.pearsonr(gt, pred)
    sn   = float(np.std(pred)) / sigma_obs
    crmse = float(np.sqrt(np.mean((pred - np.mean(pred) - gt + np.mean(gt))**2))) / sigma_obs
    taylor_data_overall[name] = (float(R), sn, crmse)

fig, ax = plt.subplots(figsize=(11, 10), facecolor='white')
taylor_diagram(ax, taylor_data_overall,
               title='Taylor Diagram — Overall (Jan–Mar 2026)\nArabian Sea SST (8.0°N, 67.0°E)')
plt.tight_layout()
plt.savefig(OUT/'plotA_taylor_diagram.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(); print("  Saved plotA_taylor_diagram.png")

# ══════════════════════════════════════════════════════════════════════════════
# 4. OVERLAID DENSITY CURVES (Plot B)
# ══════════════════════════════════════════════════════════════════════════════
print("Plot B: Overlaid Normalized Score Density Curves")

np.random.seed(42)
n_bootstrap = 1000
sample_size = 60

bootstrap_results = {}
for name, df in dfs.items():
    rmse_vals = []
    r_vals = []
    pred_all = df['predicted_avg'].values
    gt_all = df['ground_truth'].values
    n_days = len(gt_all)
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n_days, size=sample_size, replace=True)
        p_sample = pred_all[idx]
        g_sample = gt_all[idx]
        err_sample = p_sample - g_sample
        
        rmse_vals.append(np.sqrt(np.mean(err_sample**2)))
        r_val, _ = stats.pearsonr(g_sample, p_sample)
        r_vals.append(r_val)
        
    bootstrap_results[name] = {
        'rmse': np.array(rmse_vals),
        'r': np.array(r_vals)
    }

# Gather min/max across all models for normalization
all_rmse = np.concatenate([res['rmse'] for res in bootstrap_results.values()])
all_r = np.concatenate([res['r'] for res in bootstrap_results.values()])

min_rmse, max_rmse = all_rmse.min(), all_rmse.max()
min_r, max_r = all_r.min(), all_r.max()

fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')

# Reference bands and vertical lines at extreme boundaries [0, 1]
ax.axvline(0, color='black', lw=1.5, ls='-', alpha=0.8)
ax.axvline(1, color='black', lw=1.5, ls='-', alpha=0.8)
ax.axvspan(0, 0.1, alpha=0.08, color='green', label='Ideal RMSE zone')
ax.axvspan(0.9, 1.0, alpha=0.08, color='blue', label='Ideal Correlation zone')

for name, res in bootstrap_results.items():
    cfg_m = MODELS_CONFIG.get(name, {})
    col = cfg_m.get('color', 'gray')
    
    # Scale RMSE to [0.05, 0.40] -> sits toward the extreme left (0)
    if max_rmse > min_rmse:
        norm_rmse = 0.05 + (res['rmse'] - min_rmse) / (max_rmse - min_rmse) * 0.35
    else:
        norm_rmse = np.full_like(res['rmse'], 0.2)
        
    # Scale Correlation R to [0.60, 0.95] -> sits toward the extreme right (1)
    if max_r > min_r:
        norm_r = 0.60 + (res['r'] - min_r) / (max_r - min_r) * 0.35
    else:
        norm_r = np.full_like(res['r'], 0.8)
        
    # KDE for Normalized RMSE
    kde_rmse = gaussian_kde(norm_rmse, bw_method='scott')
    x_rmse = np.linspace(0, 0.5, 200)
    dens_rmse = kde_rmse(x_rmse)
    ax.plot(x_rmse, dens_rmse, lw=2.2, color=col, ls='--',
            label=f"{name} Normalized RMSE (Mean={res['rmse'].mean():.4f}°C)")
    ax.fill_between(x_rmse, dens_rmse, alpha=0.08, color=col)
    
    # KDE for Normalized Correlation (R)
    kde_r = gaussian_kde(norm_r, bw_method='scott')
    x_r = np.linspace(0.5, 1.0, 200)
    dens_r = kde_r(x_r)
    ax.plot(x_r, dens_r, lw=2.2, color=col, ls='-',
            label=f"{name} Normalized Correlation (Mean R={res['r'].mean():.4f})")
    ax.fill_between(x_r, dens_r, alpha=0.08, color=col)

ax.set_xlim(-0.05, 1.05)
ax.set_xlabel("Normalized Performance Scores (RMSE -> Left (0) | Correlation -> Right (1))", fontsize=12, fontweight='bold', labelpad=10)
ax.set_ylabel("Probability Density", fontsize=12, fontweight='bold', labelpad=10)
ax.set_title("Bootstrap Normalized Score Distributions (RMSE vs. Correlation)\n"
             "Arabian Sea SST Forecasts (Overall Q1 2026)", fontsize=13, fontweight='bold', pad=15)
ax.grid(which='major', alpha=0.15, ls='--')
ax.legend(fontsize=9.5, loc='lower center', bbox_to_anchor=(0.5, -0.28), ncol=2, framealpha=0.90)

plt.tight_layout()
plt.savefig(OUT/'plotB_error_density.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(); print("  Saved plotB_error_density.png")

# ══════════════════════════════════════════════════════════════════════════════
# 5. CLEANUP AND ZIP
# ══════════════════════════════════════════════════════════════════════════════
whitelist = ['plotA_taylor_diagram.png', 'plotB_error_density.png']
for f in list(OUT.iterdir()):
    if f.name not in whitelist:
        try:
            if f.is_file():
                f.unlink()
        except Exception:
            pass

print(f"\n{'='*60}")
print(f"COMPARISON COMPLETE -> {OUT}")
print(f"  plotA_taylor_diagram.png      - Taylor diagram (overall)")
print(f"  plotB_error_density.png       - KDE overlaid score density curves")
print(f"  Models compared: {', '.join(dfs.keys())}")

zip_path = OUT.parent / f"{OUT.name}.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in OUT.iterdir():
        if f.suffix != '.zip':
            zf.write(f, f.name)
print(f"\n  ZIP created: {zip_path}")

try:
    from google.colab import files
    files.download(str(zip_path))
    print("  Google Colab download triggered automatically.")
except Exception:
    print(f"  -> Please manually download the zip file: {zip_path}")

print(f"{'='*60}")
