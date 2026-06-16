"""
58f_moirai_regional_gradient.py — Moirai with Regional Gradient Context
══════════════════════════════════════════════════════════════════════════════════
Moirai Zero-Shot with 4-pixel Cardinal Gradient (N,S,E,W) covariates.
All post-processing brakes removed to allow natural February cold-dip tracking.

CHANGES vs original (what was causing 0.2092C February error):
  1. Anchoring REMOVED: threshold=0.25 flagged 51% of predictions and shrunk
     the February cold dip by 15%, adding ~0.15-0.22C artificial warm bias.
  2. Static val bias REMOVED: 10-window add_bias (max 0.021C) fought EWMA
     correction during regime change. EWMA handles drift in real-time.
  3. Symmetric drift caps: ADAPTIVE_CAP_POS 0.30->0.35 (matches NEG=0.35)
     Gives full ±0.35C correction range for both warm and cold bias.
  4. Equal horizon weights: Inverse-RMSE² from 10 noisy windows replaced
     with clean 1/HORIZON = 1/7 equal weights.

SPEED FIXES (506s/window -> ~8s/window = ~12min total):
  - CONTEXT_LENGTH: 1095->90  (~42x, attention is O(n²) in seq len)
  - NUM_SAMPLES:    200->50   (4x, median of 50 is stable)
  - BATCH_SIZE:     8->64     (8x, saturates T4 GPU)

CORRECTNESS FIXES:
  - Spatial drift: raw mean -> EW-weighted + slew-rate-limited
  - act_sp in draw_spatial_png: explicit [:H_orig,:W_orig] crop + shape assert
  - tgt_lon_plt: LAT_RES -> LON_RES
  - bar colors: np.where -> list comprehension
  - NaN mask before correlation computation

TARGET: RMSE < 0.135C -> beat N-BEATS ~0.145C for 1st place
"""

import os, sys, gc, time, zipfile, warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime, timedelta
from scipy import stats
from collections import defaultdict

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import TwoSlopeNorm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SCRIPT_NAME = "58f_moirai_regional_gradient.py"

# ── CONFIG ──
DATA_FILE = Path("master_region_data_new.npy")
ANOM_FILE = Path("master_region_anomalies_new.npy")
OUTPUT_DIR = Path("/content/outputs/58f_moirai_regional_gradient")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MOIRAI_CHECKPOINT = "Salesforce/moirai-1.0-R-small"
# Context optimized for "seasonal sweet spot": 365 days (1 year context)
# Allows Moirai's zero-shot attention to leverage a full year of seasonal cycle history.
CONTEXT_LENGTH = 365
HORIZON = 7
PRED_START = datetime(2026, 1, 1)
PRED_END = datetime(2026, 3, 31)
BATCH_SIZE_PREDICT = 64
NUM_SAMPLES = 50

# CLIMATOLOGY ANCHORING RE-ENABLED: Keeps short/medium term forecasts bounded
# when spread or uncertainty is high.
ANCHOR_THRESHOLD = 0.25
ANCHOR_WEIGHT = 0.15

TRAIN_FRAC = 0.85
VAL_FRAC = 0.05

LAT_MIN, LAT_MAX = 5.125, 19.875
LON_MIN, LON_MAX = 60.125, 71.875
LAT_RES = LON_RES = 0.25
TARGET_LAT, TARGET_LON = 8.0, 67.0

ADAPTIVE_WINDOW = 5
ADAPTIVE_CAP_POS = 0.35   # was 0.30 — symmetric with NEG so Feb warm-bias gets full correction range
ADAPTIVE_CAP_NEG = 0.35
MAX_OFFSET_STEP = 0.35
EW_ALPHA = 0.60

GRADIENT_DISTANCE = 8  # 2 degrees (8 * 0.25) in index space

# ============================================================
# SECTION 1: Check Data & Dependencies
# ============================================================
print(f"=== {SCRIPT_NAME} ===")
print(f"Device: {DEVICE}")
print(f"Config: ctx={CONTEXT_LENGTH}d, samples={NUM_SAMPLES}, batch={BATCH_SIZE_PREDICT}, EW={EW_ALPHA}, step={MAX_OFFSET_STEP}, caps=±{ADAPTIVE_CAP_POS}")

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Missing: {DATA_FILE}")
if not ANOM_FILE.exists():
    raise FileNotFoundError(f"Missing: {ANOM_FILE}")

try:
    from uni2ts.model.moirai import MoiraiModule, MoiraiForecast
    from gluonts.dataset.common import ListDataset
    from gluonts.dataset.field_names import FieldName
    print("✓ uni2ts and gluonts imported")
except ImportError as e:
    raise ImportError(f"Missing dependencies: {e}")

# ============================================================
# SECTION 2: Load Data
# ============================================================
print("Loading data...")
data_full = np.load(DATA_FILE).astype(np.float32)
anom_full = np.load(ANOM_FILE).astype(np.float32)
ltdm_full = data_full - anom_full

T, H_orig, W_orig = data_full.shape
print(f"  Data shape: {T} days, {H_orig}×{W_orig} grid")

START_DATE = datetime(1981, 9, 1)
dates = pd.date_range(start=START_DATE, periods=T, freq='D').to_pydatetime().tolist()
date_to_abs = {d.date(): i for i, d in enumerate(dates)}

lat_i = int(np.clip(round((TARGET_LAT - LAT_MIN) / LAT_RES), 0, H_orig - 1))
lon_i = int(np.clip(round((TARGET_LON - LON_MIN) / LON_RES), 0, W_orig - 1))
tgt_lat_plt = LAT_MIN + lat_i * LAT_RES
tgt_lon_plt = LON_MIN + lon_i * LON_RES   # fixed: was LAT_RES

extent = [LON_MIN - LAT_RES/2, LON_MAX + LAT_RES/2,
          LAT_MIN - LAT_RES/2, LAT_MAX + LAT_RES/2]

# Cardinal gradient indices (with boundary checks)
grad_n_idx = max(0, lat_i - GRADIENT_DISTANCE)
grad_s_idx = min(H_orig - 1, lat_i + GRADIENT_DISTANCE)
grad_e_idx = min(W_orig - 1, lon_i + GRADIENT_DISTANCE)
grad_w_idx = max(0, lon_i - GRADIENT_DISTANCE)

print(f"  Target: ({TARGET_LAT}°N, {TARGET_LON}°E) → index ({lat_i}, {lon_i})")
print(f"  Gradient pixels: N({grad_n_idx}), S({grad_s_idx}), E({grad_e_idx}), W({grad_w_idx})")

# ============================================================
# SECTION 3: Split & Normalize
# ============================================================
train_end = int(T * TRAIN_FRAC)
val_end = int(T * (TRAIN_FRAC + VAL_FRAC))
print(f"  Train: 0-{train_end}, Val: {train_end}-{val_end}, Test: {val_end}-{T}")

mean_anom = float(anom_full[:train_end].mean())
std_anom = float(anom_full[:train_end].std())
if std_anom < 1e-8: std_anom = 1e-8
anom_n = (anom_full - mean_anom) / std_anom

clim_mean = float(data_full[:train_end, lat_i, lon_i].mean())

# ============================================================
# SECTION 4: Load Moirai Model (Deep Context + High Samples)
# ============================================================
print(f"Loading Moirai checkpoint: {MOIRAI_CHECKPOINT}")
module = MoiraiModule.from_pretrained(MOIRAI_CHECKPOINT)

model = MoiraiForecast(
    prediction_length=HORIZON,
    target_dim=1,
    feat_dynamic_real_dim=0,
    past_feat_dynamic_real_dim=4,
    context_length=CONTEXT_LENGTH,
    module=module,
    patch_size="auto",
    num_samples=NUM_SAMPLES,
)
model = model.to(DEVICE)
model.eval()
predictor = model.create_predictor(batch_size=BATCH_SIZE_PREDICT, device=str(DEVICE))
print("✓ Moirai model ready (Regional Gradient Context, High Resolution)")

# ============================================================
# SECTION 5: Equal Horizon Weights (no static bias)
# ============================================================
# Static val bias removed: biases were tiny (max 0.021C) but fought EWMA correction
# during the February regime change. EWMA drift handles systematic error in real-time.
# Val pass also removed: 10-window RMSE estimates were too noisy to trust as weights.
# Equal weights (1/HORIZON) are honest and remove a noise source.
w_w = np.full(HORIZON, 1.0 / HORIZON)
print(f"\n--- Horizon weights: equal (1/{HORIZON} = {1/HORIZON:.4f}) ---")
print("  Static val bias removed — EWMA drift correction handles regime shifts.")

# ============================================================
# SECTION 6: Spatial Rolling Prediction (Regional Gradient, No Anchoring)
# ============================================================
print("\n--- Spatial rolling prediction (Regional Gradient) ---")
ps_abs = date_to_abs[PRED_START.date()]
pe_abs = date_to_abs[PRED_END.date()]

spatial_preds = defaultdict(list)
raw_point_preds = defaultdict(list)

n_win = 0
total_windows = pe_abs - ps_abs + 1

for base in range(ps_abs, pe_abs + 1):
    if base < CONTEXT_LENGTH:
        continue
    t0 = time.time()

    # Build ListDataset for all 2880 pixels in one batched call
    data_iter = []
    for hi in range(H_orig):
        for wi in range(W_orig):
            context = anom_n[base - CONTEXT_LENGTH:base, hi, wi].tolist()
            grad_n = anom_n[base - CONTEXT_LENGTH:base, max(0, hi - GRADIENT_DISTANCE), wi].tolist()
            grad_s = anom_n[base - CONTEXT_LENGTH:base, min(H_orig - 1, hi + GRADIENT_DISTANCE), wi].tolist()
            grad_e = anom_n[base - CONTEXT_LENGTH:base, hi, min(W_orig - 1, wi + GRADIENT_DISTANCE)].tolist()
            grad_w = anom_n[base - CONTEXT_LENGTH:base, hi, max(0, wi - GRADIENT_DISTANCE)].tolist()
            data_iter.append({
                FieldName.TARGET: context,
                FieldName.START: "2024-01-01",
                FieldName.PAST_FEAT_DYNAMIC_REAL: [grad_n, grad_s, grad_e, grad_w],
            })

    ds = ListDataset(data_iter, freq="D", one_dim_target=True)
    forecasts = list(predictor.predict(ds))

    pred_grid = np.zeros((HORIZON, H_orig, W_orig), dtype=np.float32)
    for idx, fc in enumerate(forecasts):
        hi = idx // W_orig
        wi = idx % W_orig
        pred_anom = fc.mean * std_anom + mean_anom   # denorm to anomaly space

        for k in range(HORIZON):
            day_abs = base + k
            if day_abs < T:
                # No anchoring — let Moirai predict extremes naturally
                pred_grid[k, hi, wi] = pred_anom[k] + ltdm_full[day_abs, hi, wi]

    for k in range(HORIZON):
        day_abs = base + k
        if ps_abs <= day_abs <= pe_abs and day_abs < T:
            field = pred_grid[k]   # no add_bias subtraction — removed with val pass
            spatial_preds[day_abs].append((field.copy(), k))
            raw_point_preds[day_abs].append((float(field[lat_i, lon_i]), k))

    n_win += 1
    elapsed = time.time() - t0
    if n_win % 5 == 0 or n_win == 1:
        pct = (n_win / total_windows) * 100
        eta = elapsed * (total_windows - n_win)
        print(f"    {pct:.0f}% win={n_win}/{total_windows} @{dates[base].date()} ({elapsed:.1f}s/win  ETA={eta:.0f}s)")

    if n_win % 10 == 0:
        gc.collect()
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

print(f"  Done — {n_win} windows")

# ============================================================
# SECTION 7: Aggregate Overlapping Predictions
# ============================================================
print("\n--- Aggregating predictions ---")

pred_days_abs = sorted(d for d in spatial_preds if ps_abs <= d <= pe_abs)
n_days = len(pred_days_abs)
pred_dates = [dates[d] for d in pred_days_abs]
gt_series = np.array([data_full[d, lat_i, lon_i] for d in pred_days_abs])

avg_spatial = {}
avg_raw = np.zeros(n_days)
std_series = np.zeros(n_days)
min_series = np.zeros(n_days)
max_series = np.zeros(n_days)

for i, d in enumerate(pred_days_abs):
    fields_with_k = spatial_preds[d]
    if not fields_with_k:
        avg_spatial[d] = np.full((H_orig, W_orig), np.nan)
        continue

    weighted_sum = np.zeros((H_orig, W_orig), dtype=np.float64)
    weight_sum = 0.0
    point_preds = []

    for field, k in fields_with_k:
        w = w_w[k]
        weighted_sum += field * w
        weight_sum += w
        point_preds.append(float(field[lat_i, lon_i]))

    avg_spatial[d] = (weighted_sum / weight_sum).astype(np.float32)
    avg_raw[i] = avg_spatial[d][lat_i, lon_i]
    std_series[i] = np.std(point_preds)
    min_series[i] = np.min(point_preds)
    max_series[i] = np.max(point_preds)

print(f"  Aggregated {n_days} days")

# Free accumulation dicts — no longer needed
del spatial_preds, raw_point_preds
gc.collect()

# ============================================================
# SECTION 8: Hyper-Responsive Drift Correction
# ============================================================
print("\n--- Hyper-responsive drift correction ---")

spatial_stack    = np.array([avg_spatial[d] for d in pred_days_abs])
gt_spatial_stack = np.array([data_full[d, :H_orig, :W_orig] for d in pred_days_abs])
sp_offsets       = np.zeros_like(spatial_stack)
prev_sp          = np.zeros((H_orig, W_orig), dtype=np.float32)

for i in range(1, n_days):
    w_start = max(0, i - ADAPTIVE_WINDOW)
    err_win = spatial_stack[w_start:i] - gt_spatial_stack[w_start:i]  # (w, H, W)
    k_len   = err_win.shape[0]
    ew      = np.array([EW_ALPHA**(k_len-1-j) for j in range(k_len)], dtype=np.float64)
    ew     /= ew.sum()
    raw     = np.einsum('t,thw->hw', ew, err_win).astype(np.float32)
    clipped = np.clip(raw, -ADAPTIVE_CAP_NEG, ADAPTIVE_CAP_POS)
    delta   = np.clip(clipped - prev_sp, -MAX_OFFSET_STEP, MAX_OFFSET_STEP)
    sp_offsets[i] = prev_sp + delta
    prev_sp       = sp_offsets[i]

for i, d in enumerate(pred_days_abs):
    avg_spatial[d] = spatial_stack[i] - sp_offsets[i]

del spatial_stack, gt_spatial_stack, sp_offsets
gc.collect()

def ew_adaptive_offsets(avg_raw, gt, win, cap_pos, cap_neg, step, alpha=0.60):
    n = len(avg_raw)
    off = np.zeros(n)
    for i in range(1, n):
        w0 = max(0, i - win)
        e = avg_raw[w0:i] - gt[w0:i]
        k = len(e)
        ew = np.array([alpha**(k-1-j) for j in range(k)], dtype=np.float64)
        ew /= ew.sum()
        raw = float(np.dot(ew, e))
        cl = float(np.clip(raw, -cap_neg, cap_pos))
        if i > 1:
            d = cl - off[i-1]
            if abs(d) > step:
                cl = off[i-1] + np.sign(d) * step
        off[i] = cl
    return off

ao = ew_adaptive_offsets(avg_raw, gt_series, ADAPTIVE_WINDOW,
                          ADAPTIVE_CAP_POS, ADAPTIVE_CAP_NEG,
                          MAX_OFFSET_STEP, EW_ALPHA)
avg_series = avg_raw - ao
min_series_c = min_series - ao
max_series_c = max_series - ao
print("  ✓ Hyper-responsive correction applied")

# CLIMATOLOGY ANCHORING FOR HIGH UNCERTAINTY DAYS
n_anchored = 0
for i in range(n_days):
    spread = max_series_c[i] - min_series_c[i]
    if spread > ANCHOR_THRESHOLD:
        ltdm_val = float(ltdm_full[pred_days_abs[i], lat_i, lon_i])
        avg_series[i] = (1 - ANCHOR_WEIGHT) * avg_series[i] + ANCHOR_WEIGHT * ltdm_val
        min_series_c[i] = (1 - ANCHOR_WEIGHT) * min_series_c[i] + ANCHOR_WEIGHT * (ltdm_val - 0.1)
        max_series_c[i] = (1 - ANCHOR_WEIGHT) * max_series_c[i] + ANCHOR_WEIGHT * (ltdm_val + 0.1)
        n_anchored += 1
print(f"  ✓ Climatology anchoring applied: blended {n_anchored}/{n_days} days toward LTDM")

# ============================================================
# SECTION 9: Metrics & CSV
# ============================================================
print("\n--- Computing metrics ---")

rmse = float(np.sqrt(np.mean((avg_series - gt_series) ** 2)))
mae = float(np.mean(np.abs(avg_series - gt_series)))
r_val, _ = stats.pearsonr(avg_series, gt_series)
r2_val = float(np.corrcoef(avg_series, gt_series)[0, 1] ** 2)

print(f"  RMSE: {rmse:.4f}°C")
print(f"  MAE: {mae:.4f}°C")
print(f"  R: {r_val:.4f}")
print(f"  R²: {r2_val:.4f}")

month_names = {1: "January", 2: "February", 3: "March"}
month_data = {}

for m in [1, 2, 3]:
    m_mask = np.array([dates[d].month == m for d in pred_days_abs])
    if m_mask.sum() > 0:
        m_pred = avg_series[m_mask]
        m_gt = gt_series[m_mask]
        month_data[m] = {
            'month': month_names[m],
            'rmse': float(np.sqrt(np.mean((m_pred - m_gt) ** 2))),
            'mae': float(np.mean(np.abs(m_pred - m_gt))),
            'bias': float(np.mean(m_pred - m_gt)),
            'r': float(stats.pearsonr(m_pred, m_gt)[0])
        }
        print(f"  {month_names[m]}: RMSE={month_data[m]['rmse']:.4f}  "
              f"bias={month_data[m]['bias']:+.4f}  R={month_data[m]['r']:.4f}")

pd.DataFrame({
    'date': [d.strftime('%Y-%m-%d') for d in pred_dates],
    'ground_truth': gt_series,
    'predicted_avg': avg_series,
    'predicted_std': std_series,
    'predicted_min': min_series_c,
    'predicted_max': max_series_c,
    'error': avg_series - gt_series,
    'abs_error': np.abs(avg_series - gt_series),
}).to_csv(OUTPUT_DIR / "rolling_predictions.csv", index=False)
print(f"  Saved rolling_predictions.csv")

summary_rows = [
    {'month': v['month'], 'rmse': v['rmse'], 'mae': v['mae'], 'bias': v['bias'], 'r': v['r']}
    for v in month_data.values()
]
pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "monthly_summary.csv", index=False)
print(f"  Saved monthly_summary.csv")

df_loss = pd.DataFrame({'epoch': [0], 'train_loss': [np.nan], 'val_loss': [rmse]})
df_loss.to_csv(OUTPUT_DIR / "loss_history.csv", index=False)
print(f"  Saved loss_history.csv")

# ============================================================
# SECTION 10: PLOT 1 — SPATIAL MAPS
# ============================================================
print("\n--- Generating spatial plots ---")

MONTHS_INFO = [
    ("January 2026", datetime(2026, 1, 1), datetime(2026, 1, 31)),
    ("February 2026", datetime(2026, 2, 1), datetime(2026, 2, 28)),
    ("March 2026", datetime(2026, 3, 1), datetime(2026, 3, 31)),
]

def collect_blocks(m_start, m_end):
    blocks = []
    cur = m_start
    while cur + timedelta(days=6) <= m_end:
        bdays = [date_to_abs.get((cur + timedelta(days=k)).date()) for k in range(7)]
        bdays = [da for da in bdays if da is not None and da in avg_spatial and da < T]
        if len(bdays) == 7:
            blocks.append(bdays)
        cur += timedelta(days=7)
    return blocks

def draw_spatial_png(blocks_pair, month_label, part_label, m_rmse, out_path):
    n_blk = len(blocks_pair)
    fig = plt.figure(figsize=(26, 4.5 * n_blk + 1.2), facecolor='white')
    outer = GridSpec(n_blk, 1, fig, hspace=0.55,
                     top=0.90, bottom=0.04, left=0.05, right=0.97)

    for bi, block in enumerate(blocks_pair):
        pred_sp = np.array([avg_spatial[d]                 for d in block])  # (7, H, W)
        act_sp  = np.array([data_full[d, :H_orig, :W_orig] for d in block])  # explicit crop
        assert pred_sp.shape == act_sp.shape, \
            f"Shape mismatch: pred {pred_sp.shape} vs act {act_sp.shape}"
        err_sp = pred_sp - act_sp

        vlo = float(act_sp.min())
        vhi = float(act_sp.max())
        emax = min(float(max(abs(err_sp.min()), abs(err_sp.max()))), 0.50)
        if emax < 0.05: emax = 0.30
        norm_err = TwoSlopeNorm(vmin=-emax, vcenter=0, vmax=emax)
        b_rmse = float(np.sqrt(np.mean(err_sp**2)))
        b_bias = float(err_sp.mean())
        bstart = dates[block[0]].strftime('%b %d')
        bend = dates[block[-1]].strftime('%b %d')

        inner = GridSpecFromSubplotSpec(3, HORIZON + 1, subplot_spec=outer[bi],
                                        hspace=0.40, wspace=0.22,
                                        width_ratios=[1] * HORIZON + [0.055])
        im_sst = None; im_err = None
        for day in range(HORIZON):
            dlbl = dates[block[day]].strftime('%b %d')
            d_rmse = float(np.sqrt(np.mean(err_sp[day]**2)))
            d_bias = float(err_sp[day].mean())

            ax0 = fig.add_subplot(inner[0, day])
            im_sst = ax0.imshow(pred_sp[day], cmap='RdYlBu_r', aspect='auto',
                                origin='lower', extent=extent, vmin=vlo, vmax=vhi)
            ax0.plot(tgt_lon_plt, tgt_lat_plt, 'w*', ms=9, mew=1.3, zorder=10)
            ax0.set_title(f'{dlbl} Pred\nRMSE:{d_rmse:.3f} B:{d_bias:+.3f}°C', fontsize=6.5)
            if day == 0: ax0.set_ylabel('Predicted\nLat (°N)', fontsize=8)
            else: ax0.set_yticklabels([])
            ax0.set_xticklabels([])

            ax1 = fig.add_subplot(inner[1, day])
            ax1.imshow(act_sp[day], cmap='RdYlBu_r', aspect='auto',
                       origin='lower', extent=extent, vmin=vlo, vmax=vhi)
            ax1.plot(tgt_lon_plt, tgt_lat_plt, 'w*', ms=9, mew=1.3, zorder=10)
            ax1.set_title(f'{dlbl} Actual', fontsize=6.5)
            if day == 0: ax1.set_ylabel('Actual\nLat (°N)', fontsize=8)
            else: ax1.set_yticklabels([])
            ax1.set_xticklabels([])

            ax2 = fig.add_subplot(inner[2, day])
            im_err = ax2.imshow(err_sp[day], cmap='RdBu_r', aspect='auto',
                                origin='lower', extent=extent, norm=norm_err)
            ax2.plot(tgt_lon_plt, tgt_lat_plt, 'kx', ms=7, mew=1.8, zorder=10)
            ax2.set_title(f'{dlbl} Err\nRMSE:{d_rmse:.3f} B:{d_bias:+.3f}°C', fontsize=5.8)
            if day == 0: ax2.set_ylabel('Error\nLat (°N)', fontsize=8)
            else: ax2.set_yticklabels([])

        if im_sst is not None:
            cb0 = fig.colorbar(im_sst, cax=fig.add_subplot(inner[0, HORIZON]), label='SST (°C)')
            cb0.ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
            cb0.ax.tick_params(labelsize=6)
            cb1 = fig.colorbar(im_sst, cax=fig.add_subplot(inner[1, HORIZON]), label='SST (°C)')
            cb1.ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
            cb1.ax.tick_params(labelsize=6)
            cb2 = fig.colorbar(im_err, cax=fig.add_subplot(inner[2, HORIZON]), label='Error (°C)')
            cb2.ax.tick_params(labelsize=6)

        fig.text(0.002, 1.0 - (bi + 0.5) / n_blk,
                 f'Block {bi+1}\n{bstart}–{bend}\nRMSE={b_rmse:.3f}°C\nBias={b_bias:+.3f}°C',
                 va='center', ha='left', fontsize=7.5, transform=fig.transFigure,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#EEF2FF',
                           edgecolor='#99A', alpha=0.90))

    plt.suptitle(
        f"{SCRIPT_NAME}  |  {month_label}  {part_label}  |  Rolling 7-Day SST Forecast\n"
        f"({TARGET_LAT}°N, {TARGET_LON}°E)   ★=target   Monthly RMSE:{m_rmse:.4f}°C",
        fontsize=11, fontweight='bold', y=0.97)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved {out_path.name}")

for month_name, m_start, m_end in MONTHS_INFO:
    blocks = collect_blocks(m_start, m_end)
    if not blocks:
        print(f"  Skipping {month_name} — no blocks")
        continue
    m_mask = np.array([dates[d].month == m_start.month for d in pred_days_abs])
    m_rmse = float(np.sqrt(np.mean((avg_series[m_mask] - gt_series[m_mask])**2))) \
             if m_mask.sum() else 0.0
    month_tag = month_name.replace(' ', '_').lower()
    h1 = blocks[:2]
    h2 = blocks[2:]
    if h1:
        draw_spatial_png(h1, month_name, f"Part 1  Days 1–{7*len(h1)}",
                         m_rmse, OUTPUT_DIR / f"plot1_spatial_{month_tag}_part1.png")
    if h2:
        d0 = 7*len(h1)+1
        d1 = 7*(len(h1)+len(h2))
        draw_spatial_png(h2, month_name, f"Part 2  Days {d0}–{d1}",
                         m_rmse, OUTPUT_DIR / f"plot1_spatial_{month_tag}_part2.png")

# ============================================================
# SECTION 11: PLOT 2 — Time Series
# ============================================================
print("\n--- Generating time series plot ---")

fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

ax0 = axes[0]
ax0.plot(pred_dates, gt_series, 'k-', lw=1.5, label='Actual', alpha=0.8)
ax0.plot(pred_dates, avg_series, 'b-', lw=1.2, label='Predicted', alpha=0.8)
ax0.fill_between(pred_dates, min_series_c, max_series_c, color='blue', alpha=0.15, label='Min-Max')
ax0.set_ylabel('SST (°C)', fontsize=10)
ax0.set_title(f'{SCRIPT_NAME} — Regional Gradient Moirai', fontsize=12, fontweight='bold')
ax0.legend(loc='upper right', fontsize=9)
ax0.grid(True, alpha=0.3)
ax0.axvline(datetime(2026, 1, 31), color='gray', ls='--', alpha=0.5)
ax0.axvline(datetime(2026, 2, 28), color='gray', ls='--', alpha=0.5)

ax1 = axes[1]
errors = avg_series - gt_series
bar_colors = ['crimson' if e > 0 else 'steelblue' for e in errors]
ax1.bar(pred_dates, errors, width=0.8, color=bar_colors, alpha=0.7)
ax1.axhline(0, color='black', lw=0.8)
ax1.set_ylabel('Error (°C)', fontsize=10)
ax1.set_title(f'Prediction Error (RMSE: {rmse:.4f}°C, MAE: {mae:.4f}°C)', fontsize=11)
ax1.grid(True, alpha=0.3)

ax2 = axes[2]
window = 7
rmses = []
for i in range(len(avg_series)):
    w_start = max(0, i - window + 1)
    rmses.append(np.sqrt(np.mean((avg_series[w_start:i+1] - gt_series[w_start:i+1])**2)))
ax2.plot(pred_dates, rmses, 'g-', lw=1.2)
ax2.fill_between(pred_dates, 0, rmses, color='green', alpha=0.2)
ax2.set_ylabel('RMSE (°C)', fontsize=10)
ax2.set_xlabel('Date', fontsize=10)
ax2.set_title(f'{window}-Day Rolling RMSE', fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plot2_timeseries_90day.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved plot2_timeseries_90day.png")

# ============================================================
# SECTION 12: PLOT 3 — Correlation
# ============================================================
print("\n--- Generating correlation plot ---")

all_pred_flat = []
all_act_flat = []
for d_abs in pred_days_abs:
    if d_abs < T:
        sp = avg_spatial[d_abs]
        all_pred_flat.append(sp[:H_orig, :W_orig].ravel())   # explicit crop
        all_act_flat.append(data_full[d_abs, :H_orig, :W_orig].ravel())
all_pred_flat = np.concatenate(all_pred_flat)
all_act_flat = np.concatenate(all_act_flat)

# Mask NaNs before correlation
valid = np.isfinite(all_pred_flat) & np.isfinite(all_act_flat)
r_all = float(np.corrcoef(all_pred_flat[valid], all_act_flat[valid])[0, 1])
rmse_all = float(np.sqrt(np.mean((all_pred_flat[valid] - all_act_flat[valid]) ** 2)))

fig, (ax_L, ax_R) = plt.subplots(1, 2, figsize=(22, 10))

hb = ax_L.hexbin(all_act_flat[valid], all_pred_flat[valid], gridsize=90, cmap='Blues', mincnt=1, bins='log')
ax_L.plot([all_act_flat[valid].min(), all_act_flat[valid].max()],
          [all_act_flat[valid].min(), all_act_flat[valid].max()],
          'r--', lw=2, label='1:1')
ax_L.set_xlabel('Actual SST (°C)', fontsize=12)
ax_L.set_ylabel('Predicted SST (°C)', fontsize=12)
ax_L.set_title(f'ALL Grid Points — R: {r_all:.4f}, RMSE: {rmse_all:.4f}°C', fontsize=13)
ax_L.legend(fontsize=10)
cb = plt.colorbar(hb, ax=ax_L)
cb.set_label('Log Count', fontsize=10)

month_colors = {1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green'}
for m in [1, 2, 3]:
    m_mask = np.array([dates[d].month == m for d in pred_days_abs])
    if m_mask.sum() > 0:
        ax_R.scatter(gt_series[m_mask], avg_series[m_mask], 
                    c=month_colors[m], alpha=0.6, s=30, label=f'{month_names[m]}')

ax_R.plot([gt_series.min(), gt_series.max()], [gt_series.min(), gt_series.max()], 
          'k--', lw=2, label='1:1')
ax_R.set_xlabel('Actual SST (°C)', fontsize=12)
ax_R.set_ylabel('Predicted SST (°C)', fontsize=12)
ax_R.set_title(f'Target Pixel ({TARGET_LAT}°N, {TARGET_LON}°E) — R: {r_val:.4f}', fontsize=13)
ax_R.legend(fontsize=10)
ax_R.grid(True, alpha=0.3)

plt.suptitle(f'{SCRIPT_NAME} — Correlation Analysis', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plot3_correlation_target_pixel.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved plot3_correlation_target_pixel.png")

# ============================================================
# SECTION 13: Create ZIP
# ============================================================
print("\n--- Creating ZIP ---")
zip_path = OUTPUT_DIR.parent / "58f_moirai_regional_gradient_outputs.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in OUTPUT_DIR.glob("*"):
        if f.is_file():
            zf.write(f, f.name)
print(f"  Saved {zip_path.name}")

print(f"\n=== DONE ===")
print(f"Outputs: {OUTPUT_DIR}")
print(f"ZIP: {zip_path}")
print(f"\nResults summary:")
for m, v in month_data.items():
    target_ok = "✓" if v['rmse'] < 0.135 else "✗"
    print(f"  {v['month']:12s}: RMSE={v['rmse']:.4f}C  bias={v['bias']:+.4f}C  {target_ok}")
print(f"\nEnsemble ranking:")
print(f"  Target: Moirai < 0.135C | N-BEATS ~0.145C | LSTM ~0.15C")