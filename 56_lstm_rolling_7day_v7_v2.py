"""
56_lstm_rolling_7day.py  —  Rolling 7-Day LSTM SST Forecast  (FINAL)
═══════════════════════════════════════════════════════════════════════════════
FIX (v2 → fixed): vp_pt / va_pt were referenced before assignment inside
compute_corrections().  Added the two missing slice lines immediately after
vp_arr / va_arr are built.
═══════════════════════════════════════════════════════════════════════════════
"""

import os, time, zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime, timedelta
from scipy import stats
from collections import defaultdict

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import TwoSlopeNorm
import matplotlib.patches as mpatches

torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    SCRIPT_NAME = os.path.basename(__file__)
except NameError:
    SCRIPT_NAME = "56_lstm_rolling_7day_v7.py"

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
DATA_FILE  = Path("/content/drive/MyDrive/master_region_data_new.npy")
ANOM_FILE  = Path("/content/drive/MyDrive/master_region_anomalies_new.npy")
OUTPUT_DIR = Path("/content/outputs/56_lstm_rolling_7day")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_BEST  = OUTPUT_DIR / "model_best.pt"

LAT_MIN, LAT_MAX = 5.125, 19.875
LON_MIN, LON_MAX = 60.125, 71.875
LAT_RES = LON_RES = 0.25
TARGET_LAT, TARGET_LON = 8.0, 67.0

PRED_START_DATE = datetime(2026, 1, 1)
PRED_END_DATE   = datetime(2026, 3, 31)
START_DATE      = datetime(1981, 9, 1)

SEQ_LEN        = 90
HORIZON        = 7
BATCH_SIZE     = 4
EPOCHS         = 35
HIDDEN_DIM     = 96
NUM_LAYERS     = 2
WEIGHT_DECAY   = 1e-4
INPUT_CHANNELS = 4
PATIENCE       = 10
TRAIN_FRAC     = 0.85
VAL_FRAC       = 0.05
ADAPTIVE_WINDOW = 7

# Adaptive offset config
ADAPTIVE_CAP_POS  = 0.30
ADAPTIVE_CAP_NEG  = 0.35
MAX_OFFSET_STEP   = 0.10
SPATIAL_ADAPTIVE_WINDOW = 5
SPATIAL_CAP = 0.25

def latlon_to_idx(lat, lon, h, w):
    ri = int(np.clip(round((lat-LAT_MIN)/LAT_RES), 0, h-1))
    ci = int(np.clip(round((lon-LON_MIN)/LON_RES), 0, w-1))
    print(f"  ({lat}N,{lon}E)->idx({ri},{ci})->({LAT_MIN+ri*LAT_RES:.3f}N,{LON_MIN+ci*LON_RES:.3f}E)")
    return ri, ci

# ═══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE OFFSET HELPER
# ═══════════════════════════════════════════════════════════════════════════════
def build_adaptive_offsets(avg_raw, gt_series, window, cap_pos, cap_neg, max_step,
                            ew_alpha=0.85):
    """
    Exponentially-weighted adaptive offset within a causal window-day lookback.
    Most recent day gets weight 1.0; day W ago gets weight ew_alpha^(W-1).
    """
    n_days = len(avg_raw)
    adaptive_offsets = np.zeros(n_days)
    cap_pos_hits = 0; cap_neg_hits = 0; slew_hits = 0

    for i in range(1, n_days):
        w_start = max(0, i - window)
        window_errors = avg_raw[w_start:i] - gt_series[w_start:i]
        k = len(window_errors)
        ew = np.array([ew_alpha ** (k - 1 - j) for j in range(k)], dtype=np.float64)
        ew /= ew.sum()
        offset_raw = float(np.dot(ew, window_errors))
        offset_clipped = float(np.clip(offset_raw, -cap_neg, cap_pos))
        if offset_clipped >= cap_pos:  cap_pos_hits += 1
        if offset_clipped <= -cap_neg: cap_neg_hits += 1
        if i > 1:
            delta = offset_clipped - adaptive_offsets[i - 1]
            if abs(delta) > max_step:
                offset_clipped = adaptive_offsets[i - 1] + np.sign(delta) * max_step
                slew_hits += 1
        adaptive_offsets[i] = offset_clipped

    return adaptive_offsets, cap_pos_hits, cap_neg_hits, slew_hits

print(f"=== {SCRIPT_NAME} ===  SEQ={SEQ_LEN} H={HORIZON} HIDDEN={HIDDEN_DIM} Device={DEVICE}")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA
# ═══════════════════════════════════════════════════════════════════════════════
data_full = np.load(DATA_FILE).astype(np.float32)
anom_full = np.load(ANOM_FILE).astype(np.float32)
ltdm_full = data_full - anom_full
T, H_orig, W_orig = data_full.shape
dates       = pd.date_range(start=START_DATE, periods=T, freq='D').to_pydatetime().tolist()
lat_i,lon_i = latlon_to_idx(TARGET_LAT, TARGET_LON, H_orig, W_orig)
date_to_abs = {d.date():i for i,d in enumerate(dates)}

extent = [LON_MIN-LAT_RES/2, LON_MAX+LAT_RES/2,
          LAT_MIN-LAT_RES/2, LAT_MAX+LAT_RES/2]
tgt_lon_plt = LON_MIN + lon_i*LON_RES
tgt_lat_plt = LAT_MIN + lat_i*LAT_RES

print(f"  Data:{data_full.shape} | {dates[0].date()}->{dates[-1].date()}")
print(f"  Spatial extent:{extent}  target:({tgt_lat_plt:.3f}N,{tgt_lon_plt:.3f}E)")

anom_pad = np.pad(anom_full,((0,0),(0,0),(0,2)),mode='edge')
ltdm_pad = np.pad(ltdm_full,((0,0),(0,0),(0,2)),mode='edge')
H_pad, W_pad = 60, 50
lat_grid = np.repeat(np.linspace(0,1,H_pad,dtype=np.float32).reshape(H_pad,1)
                     .repeat(W_pad,1)[np.newaxis], T, axis=0)
lon_grid = np.repeat(np.linspace(0,1,W_pad,dtype=np.float32).reshape(1,W_pad)
                     .repeat(H_pad,0)[np.newaxis], T, axis=0)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. SPLIT & NORMALIZE
# ═══════════════════════════════════════════════════════════════════════════════
train_end = int(T*TRAIN_FRAC)
val_end   = int(T*(TRAIN_FRAC+VAL_FRAC))
print(f"  Train->{dates[train_end-1].date()} | Val->{dates[val_end-1].date()}")

def get_norm(arr):
    m=arr.mean(0).astype(np.float32); s=arr.std(0).astype(np.float32)
    s[s==0]=1e-8; return m,s

mean_anom,std_anom = get_norm(anom_pad[:train_end])
mean_ltdm,std_ltdm = get_norm(ltdm_pad[:train_end])
anom_n = ((anom_pad-mean_anom)/std_anom).astype(np.float32)
ltdm_n = ((ltdm_pad-mean_ltdm)/std_ltdm).astype(np.float32)

sl={'train':slice(0,train_end),'val':slice(train_end,val_end),'test':slice(val_end,T)}
combo     = {k:np.stack([anom_n[v],ltdm_n[v],lat_grid[v],lon_grid[v]],1).astype(np.float32)
             for k,v in sl.items()}
anom_targ = {k:anom_n[v] for k,v in sl.items()}
clim_mean = float(data_full[:train_end,lat_i,lon_i].mean())
print(f"  Clim mean={clim_mean:.3f}°C")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. DATASET & MODEL
# ═══════════════════════════════════════════════════════════════════════════════
class PixelWiseDeltaDataset(Dataset):
    def __init__(self,combo,target,seq,horizon):
        self.combo=combo; self.target=target; self.seq=seq; self.horizon=horizon
    def __len__(self): return len(self.combo)-self.seq-self.horizon+1
    def __getitem__(self,idx):
        X=self.combo[idx:idx+self.seq]; la=self.target[idx+self.seq-1]
        Yd=self.target[idx+self.seq:idx+self.seq+self.horizon]-la[np.newaxis]
        return torch.from_numpy(X.copy()),torch.from_numpy(Yd.copy()),torch.from_numpy(la.copy())

class LevelConditionedLSTM(nn.Module):
    def __init__(self,in_feat,hid,out_h):
        super().__init__()
        self.lstm=nn.LSTM(in_feat,hid,num_layers=NUM_LAYERS,batch_first=True)
        self.head=nn.Linear(hid+1,out_h)
    def forward(self,x,la):
        B,Seq,C,Hg,Wg=x.shape
        x=x.permute(0,3,4,1,2).reshape(B*Hg*Wg,Seq,C)
        _,(hn,_)=self.lstm(x)
        out=self.head(torch.cat([hn[-1],la.reshape(B*Hg*Wg,1)],1))
        return out.view(B,Hg,Wg,-1).permute(0,3,1,2)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'#'*55}\n# TRAINING\n{'#'*55}")
kw=dict(num_workers=2,pin_memory=True)
train_ds=PixelWiseDeltaDataset(combo['train'],anom_targ['train'],SEQ_LEN,HORIZON)
val_ds  =PixelWiseDeltaDataset(combo['val'],  anom_targ['val'],  SEQ_LEN,HORIZON)
trL=DataLoader(train_ds,BATCH_SIZE,shuffle=True, drop_last=True,**kw)
vaL=DataLoader(val_ds,  BATCH_SIZE,shuffle=False,**kw)
model=LevelConditionedLSTM(INPUT_CHANNELS,HIDDEN_DIM,HORIZON).to(DEVICE)
print(f"  Params:{sum(p.numel() for p in model.parameters()):,}")
opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=WEIGHT_DECAY)
lossfn=nn.HuberLoss(delta=0.5)
sc=torch.amp.GradScaler('cuda') if DEVICE.type=='cuda' else None
sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=1.5e-3,
      total_steps=len(trL)*EPOCHS,pct_start=0.3)
best_val=float('inf'); early_cnt=0; loss_hist={'train':[],'val':[]}
for ep in range(EPOCHS):
    model.train(); tr=0.; t0=time.time()
    for X,Yd,la in trL:
        X,Yd,la=X.to(DEVICE),Yd.to(DEVICE),la.to(DEVICE); opt.zero_grad()
        if sc:
            with torch.autocast('cuda',dtype=torch.float16): loss=lossfn(model(X,la),Yd)
            sc.scale(loss).backward(); sc.step(opt); sc.update()
        else:
            loss=lossfn(model(X,la),Yd); loss.backward(); opt.step()
        sched.step(); tr+=loss.item()
    model.eval(); vl=0.
    with torch.no_grad():
        for X,Yd,la in vaL:
            X,Yd,la=X.to(DEVICE),Yd.to(DEVICE),la.to(DEVICE)
            vl+=lossfn(model(X,la),Yd).item()
    atr=tr/len(trL); avl=vl/max(1,len(vaL))
    loss_hist['train'].append(atr); loss_hist['val'].append(avl)
    print(f"  Ep{ep+1:02d}/{EPOCHS} Tr:{atr:.6f} Val:{avl:.6f} ({time.time()-t0:.1f}s)")
    if avl<best_val:
        best_val=avl; torch.save(model.state_dict(),CKPT_BEST)
        early_cnt=0; print(f"    **Best:{best_val:.6f}")
    else:
        early_cnt+=1
        if early_cnt>=PATIENCE: print("  Early stop."); break
pd.DataFrame(loss_hist).to_csv(OUTPUT_DIR/"loss_history.csv",index=False)
model.load_state_dict(torch.load(CKPT_BEST,weights_only=True))
print("  Loaded best weights.")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. VALIDATION-SET CORRECTIONS
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*55}\nCORRECTIONS\n{'='*55}")

def compute_corrections(model, val_ds, horizon):
    val_start=train_end+SEQ_LEN; vp_list=[]
    model.eval()
    with torch.no_grad():
        for X,_,last_a in DataLoader(val_ds,batch_size=1,shuffle=False):
            X,last_a=X.to(DEVICE),last_a.to(DEVICE)
            delta=model(X,last_a).squeeze(0).cpu().numpy()
            la_np=last_a.squeeze(0).cpu().numpy()
            vp_list.append(delta+la_np[np.newaxis])
    converted,actuals=[],[]
    for i,pred_n in enumerate(vp_list):
        base=val_start+i
        if base+horizon>val_end: break
        pa=(pred_n*std_anom)+mean_anom
        ps=pa[:,:,:W_orig]+ltdm_full[base:base+horizon]
        converted.append(ps); actuals.append(data_full[base:base+horizon])
    n=min(len(converted),len(actuals))
    if n==0:
        return (np.zeros(horizon), np.ones(horizon), np.zeros(horizon),
                np.zeros((horizon,H_orig,W_orig)), 1.0)

    vp_arr=np.array(converted[:n])   # (n, horizon, H_orig, W_orig)
    va_arr=np.array(actuals[:n])     # (n, horizon, H_orig, W_orig)

    # ── FIX: define target-pixel slices BEFORE they are used below ──
    vp_pt = vp_arr[:, :, lat_i, lon_i]   # (n, horizon)
    va_pt = va_arr[:, :, lat_i, lon_i]   # (n, horizon)

    # Seasonally-stratified additive bias
    val_start_abs = train_end + SEQ_LEN
    cool_months   = {11,12,1,2,3,4}
    cool_mask     = np.array([dates[val_start_abs+i].month in cool_months
                               for i in range(n)])
    if cool_mask.sum() >= HORIZON * 3:
        add_bias = np.mean((vp_pt - va_pt)[cool_mask], axis=0)
        print(f"  add_bias (cool-season n={cool_mask.sum()}): {np.round(add_bias,4)}")
    else:
        add_bias = np.mean(vp_pt - va_pt, axis=0)
        print(f"  add_bias (all-season n={n}): {np.round(add_bias,4)}")

    actual_mean=va_pt.mean(axis=0)
    scale=np.ones(horizon,dtype=np.float32)
    for k in range(horizon):
        vp_c=(vp_pt[:,k]-add_bias[k])-actual_mean[k]
        va_c=va_pt[:,k]-actual_mean[k]
        if va_c.std()<1e-6: continue
        sl2,_,r,_,_=stats.linregress(va_c,vp_c)
        if r**2>0.25 and 0.4<sl2<2.5: scale[k]=float(sl2)

    # Full-grid 2D spatial bias — median robust to outliers
    spatial_bias2d=np.median(vp_arr-va_arr,axis=0).astype(np.float32)

    # Val-set slope for variance inflation
    vp_flat=vp_pt.flatten(); va_flat=va_pt.flatten()
    val_slope_vi,_,r_vi,_,_=stats.linregress(va_flat,vp_flat)
    val_slope_vi=float(val_slope_vi)

    print(f"  add_bias   :{np.round(add_bias,4)}")
    print(f"  scale      :{np.round(scale,4)}")
    print(f"  actual_mean:{np.round(actual_mean,3)}")
    print(f"  spatial_bias2d mean/step:{np.round(spatial_bias2d.mean(axis=(1,2)),4)}")
    print(f"  val_slope_vi={val_slope_vi:.4f}  (R={r_vi:.3f})")
    return add_bias, scale, actual_mean, spatial_bias2d, val_slope_vi

add_bias,scale,actual_mean,spatial_bias2d,val_slope_vi=compute_corrections(model,val_ds,HORIZON)

# ═══════════════════════════════════════════════════════════════════════════════
# 6. ROLLING PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*55}\nROLLING PREDICTION\n{'='*55}")

pred_start_abs=date_to_abs.get(PRED_START_DATE.date(),
    next(i for i,d in enumerate(dates) if d>=PRED_START_DATE))
pred_end_abs=date_to_abs.get(PRED_END_DATE.date(),
    next(i for i,d in enumerate(dates) if d>=PRED_END_DATE))

raw_point_preds=defaultdict(list)
spatial_preds  =defaultdict(list)

rmse_by_horizon=np.array([0.128438, 0.190926, 0.231797, 0.262054,
                           0.286933, 0.308568, 0.324445])
w_inv_rmse2=(1.0 / (rmse_by_horizon**2))
w_inv_rmse2=w_inv_rmse2/w_inv_rmse2.sum()
print(f"  Horizon weights (inv RMSE²): {np.round(w_inv_rmse2,4)}")

model.eval(); n_windows=0
with torch.no_grad():
    for t in range(pred_start_abs,pred_end_abs+1):
        if t-SEQ_LEN<0: continue
        X=np.stack([anom_n[t-SEQ_LEN:t],ltdm_n[t-SEQ_LEN:t],
                    lat_grid[t-SEQ_LEN:t],lon_grid[t-SEQ_LEN:t]],axis=1)
        last_a=anom_n[t-1]
        delta=model(torch.from_numpy(X[np.newaxis]).to(DEVICE),
                    torch.from_numpy(last_a[np.newaxis]).to(DEVICE)
                   ).squeeze(0).cpu().numpy()
        pred_norm=delta+last_a[np.newaxis]
        pred_anom=(pred_norm*std_anom)+mean_anom
        pred_anom_orig=pred_anom[:,:,:W_orig]
        for k in range(HORIZON):
            day_abs=t+k
            if day_abs>pred_end_abs or day_abs>=T: break
            pred_sst=pred_anom_orig[k]+ltdm_full[day_abs]
            pt_val=float(pred_sst[lat_i,lon_i])
            raw_point_preds[day_abs].append((pt_val, k))
            spatial_preds[day_abs].append((pred_sst-spatial_bias2d[k], k))
        n_windows+=1
        if n_windows%25==0:
            pct=(t-pred_start_abs)/(pred_end_abs-pred_start_abs)*100
            print(f"    {pct:.0f}% win{n_windows} @{dates[t].date()}")

print(f"  Total windows:{n_windows}")

pred_days_abs=sorted(d for d in raw_point_preds if pred_start_abs<=d<=pred_end_abs)
n_days=len(pred_days_abs)
pred_dates=[dates[d] for d in pred_days_abs]
gt_series=np.array([data_full[d,lat_i,lon_i] for d in pred_days_abs])

avg_raw=np.zeros(n_days); std_series=np.zeros(n_days)
min_series=np.zeros(n_days); max_series=np.zeros(n_days)
n_overlaps=np.zeros(n_days,dtype=int)
avg_spatial={}

for i,d in enumerate(pred_days_abs):
    preds_with_k=raw_point_preds[d]
    if not preds_with_k:
        avg_raw[i]=np.nan; std_series[i]=0
        min_series[i]=np.nan; max_series[i]=np.nan
        n_overlaps[i]=0; avg_spatial[d]=np.zeros_like(spatial_bias2d[0])
        continue

    preds_corrected=[]
    spatial_corrected=[]
    weights=[]
    for pt_val,k in preds_with_k:
        pt_corr=pt_val-add_bias[k]
        preds_corrected.append(pt_corr)
        weights.append(w_inv_rmse2[k])
    for sp_field,k in spatial_preds[d]:
        spatial_corrected.append(sp_field-add_bias[k])

    avg_raw[i]=np.average(preds_corrected,weights=weights)
    min_series[i]=np.min(preds_corrected)
    max_series[i]=np.max(preds_corrected)
    std_series[i]=np.std(preds_corrected)
    n_overlaps[i]=len(preds_with_k)
    avg_spatial[d]=np.average(spatial_corrected,axis=0,weights=weights)

# ═══════════════════════════════════════════════════════════════════════════════
# 7. ADAPTIVE DRIFT CORRECTION
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n  Adaptive settings: window={ADAPTIVE_WINDOW}d, "
      f"cap=({-ADAPTIVE_CAP_NEG:.2f}, +{ADAPTIVE_CAP_POS:.2f})°C, "
      f"max_step={MAX_OFFSET_STEP:.2f}°C, spatial_win={SPATIAL_ADAPTIVE_WINDOW}d, "
      f"spatial_cap=±{SPATIAL_CAP:.2f}°C, EW α=0.85")

adaptive_offsets, cap_pos_hits, cap_neg_hits, slew_hits = build_adaptive_offsets(
    avg_raw, gt_series, ADAPTIVE_WINDOW, ADAPTIVE_CAP_POS, ADAPTIVE_CAP_NEG, MAX_OFFSET_STEP
)

print(f"  Adaptive offsets: min={adaptive_offsets.min():+.4f}  "
      f"max={adaptive_offsets.max():+.4f}  mean={adaptive_offsets.mean():+.4f}°C")
print(f"  Cap hits: +{ADAPTIVE_CAP_POS}={cap_pos_hits}, "
      f"-{ADAPTIVE_CAP_NEG}={cap_neg_hits}, slew={slew_hits}")

avg_series  =avg_raw - adaptive_offsets
min_series_c=min_series - adaptive_offsets
max_series_c=max_series - adaptive_offsets

# Variance inflation
VI_FACTOR = 1.0
if val_slope_vi < 0.99:
    VI_FACTOR = min(1.0 / val_slope_vi, 1.08)
    pred_center = float(np.mean(avg_series))
    avg_series   = pred_center + (avg_series   - pred_center) * VI_FACTOR
    min_series_c = pred_center + (min_series_c - pred_center) * VI_FACTOR
    max_series_c = pred_center + (max_series_c - pred_center) * VI_FACTOR
    print(f"  Variance inflation: val_slope={val_slope_vi:.4f} → VI_FACTOR={VI_FACTOR:.4f}")
else:
    print(f"  Variance inflation: val_slope={val_slope_vi:.4f} — no inflation needed")

# Scalar spatial correction (CRITICAL — not per-pixel)
spatial_stack    = np.array([avg_spatial[d] for d in pred_days_abs])
gt_spatial_stack = np.array([data_full[d,:H_orig,:W_orig] for d in pred_days_abs])
sp_scalar_offsets = np.zeros(n_days)
for i in range(1, n_days):
    w_start = max(0, i - SPATIAL_ADAPTIVE_WINDOW)
    sp_scalar_offsets[i] = float(np.clip(
        np.mean(spatial_stack[w_start:i] - gt_spatial_stack[w_start:i]),
        -SPATIAL_CAP, SPATIAL_CAP))
for i, d in enumerate(pred_days_abs):
    avg_spatial[d] = spatial_stack[i] - sp_scalar_offsets[i]

rmse_90=float(np.sqrt(np.mean((avg_series-gt_series)**2)))
mae_90 =float(np.mean(np.abs(avg_series-gt_series)))
r_val  =float(np.corrcoef(avg_series,gt_series)[0,1])
r2_val =r_val**2
slope_pt,interc_pt,_,_,_=stats.linregress(gt_series,avg_series)
print(f"\n  RMSE={rmse_90:.4f}°C  MAE={mae_90:.4f}°C  R²={r2_val:.4f}  R={r_val:.4f}")
print(f"  Slope={slope_pt:.4f}  Intercept={interc_pt:.4f}°C")

month_names={1:'Jan',2:'Feb',3:'Mar'}
month_mask ={m:np.array([d.month==m for d in pred_dates]) for m in [1,2,3]}
m_cols_ts  ={1:'#2874A6',2:'#1E8449',3:'#C0392B'}

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def set_quarter_yticks(ax,ylo,yhi,margin=0.35):
    lo=np.floor((ylo-margin)*4)/4; hi=np.ceil((yhi+margin)*4)/4
    ax.set_ylim(lo,hi)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

def style_ax(ax):
    ax.grid(which='major',alpha=0.18,ls='--',lw=0.6)
    ax.grid(which='minor',alpha=0.07,ls=':',lw=0.4)
    for sp in ax.spines.values(): sp.set_linewidth(1.1)

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1: SPATIAL
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*55}\nPLOT 1: SPATIAL\n{'='*55}")

MONTHS_INFO=[
    ("January 2026",  datetime(2026,1,1),  datetime(2026,1,31)),
    ("February 2026", datetime(2026,2,1),  datetime(2026,2,28)),
    ("March 2026",    datetime(2026,3,1),  datetime(2026,3,31)),
]

def collect_blocks(m_start,m_end):
    blocks=[]; cur=m_start
    while cur<=m_end:
        days_left=(m_end-cur).days+1
        n_in_block=min(7,days_left)
        bdays=[date_to_abs.get((cur+timedelta(days=k)).date()) for k in range(n_in_block)]
        bdays=[da for da in bdays if da is not None and da in avg_spatial and da<T]
        if bdays: blocks.append(bdays)
        cur+=timedelta(days=7)
    return blocks

def draw_spatial_png(blocks_pair, month_label, part_label, m_rmse, out_path):
    n_blk=len(blocks_pair)
    COL_W = 3.4
    fig_w = HORIZON * COL_W + 1.2
    fig=plt.figure(figsize=(fig_w, 4.5*n_blk+1.2), facecolor='white')
    outer=GridSpec(n_blk,1,fig,hspace=0.55,
                   top=0.90,bottom=0.04,left=0.05,right=0.97)

    for bi,block in enumerate(blocks_pair):
        n_days_blk=len(block)
        pred_sp=np.array([avg_spatial[d] for d in block])
        act_sp =np.array([data_full[d]   for d in block])
        err_sp =pred_sp-act_sp

        vlo=float(act_sp.min()); vhi=float(act_sp.max())
        emax=min(float(max(abs(err_sp.min()),abs(err_sp.max()))),0.50)
        if emax<0.05: emax=0.30
        norm_err=TwoSlopeNorm(vmin=-emax,vcenter=0,vmax=emax)

        b_rmse=float(np.sqrt(np.mean(err_sp**2)))
        b_bias=float(err_sp.mean())
        bstart=dates[block[0]].strftime('%b %d')
        bend  =dates[block[-1]].strftime('%b %d')

        inner=GridSpecFromSubplotSpec(3, HORIZON+1, subplot_spec=outer[bi],
                                      hspace=0.40, wspace=0.22,
                                      width_ratios=[1]*HORIZON+[0.055])

        im_sst=None; im_err=None
        for day in range(n_days_blk):
            dlbl  =dates[block[day]].strftime('%b %d')
            d_rmse=float(np.sqrt(np.mean(err_sp[day]**2)))
            d_bias=float(err_sp[day].mean())

            ax0=fig.add_subplot(inner[0,day])
            im_sst=ax0.imshow(pred_sp[day],cmap='RdYlBu_r',aspect='auto',
                              origin='lower',extent=extent,vmin=vlo,vmax=vhi)
            ax0.plot(tgt_lon_plt,tgt_lat_plt,'w*',ms=9,mew=1.3,zorder=10)
            ax0.set_title(f'{dlbl} Pred\nRMSE:{d_rmse:.3f} B:{d_bias:+.3f}°C',fontsize=6.5)
            if day==0: ax0.set_ylabel('Predicted\nLat (°N)',fontsize=8)
            else: ax0.set_yticklabels([])
            ax0.set_xticklabels([])

            ax1=fig.add_subplot(inner[1,day])
            ax1.imshow(act_sp[day],cmap='RdYlBu_r',aspect='auto',
                       origin='lower',extent=extent,vmin=vlo,vmax=vhi)
            ax1.plot(tgt_lon_plt,tgt_lat_plt,'w*',ms=9,mew=1.3,zorder=10)
            ax1.set_title(f'{dlbl} Actual',fontsize=6.5)
            if day==0: ax1.set_ylabel('Actual\nLat (°N)',fontsize=8)
            else: ax1.set_yticklabels([])
            ax1.set_xticklabels([])

            ax2=fig.add_subplot(inner[2,day])
            im_err=ax2.imshow(err_sp[day],cmap='RdBu_r',aspect='auto',
                              origin='lower',extent=extent,norm=norm_err)
            ax2.plot(tgt_lon_plt,tgt_lat_plt,'kx',ms=7,mew=1.8,zorder=10)
            ax2.set_title(f'{dlbl} Err\nRMSE:{d_rmse:.3f} B:{d_bias:+.3f}°C',fontsize=5.8)
            if day==0: ax2.set_ylabel('Error\nLat (°N)',fontsize=8)
            else: ax2.set_yticklabels([])

        if im_sst is not None:
            cb0=fig.colorbar(im_sst,cax=fig.add_subplot(inner[0,HORIZON]),label='SST (°C)')
            cb0.ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
            cb0.ax.tick_params(labelsize=6)
            cb1=fig.colorbar(im_sst,cax=fig.add_subplot(inner[1,HORIZON]),label='SST (°C)')
            cb1.ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
            cb1.ax.tick_params(labelsize=6)
            cb2=fig.colorbar(im_err,cax=fig.add_subplot(inner[2,HORIZON]),label='Error (°C)')
            cb2.ax.tick_params(labelsize=6)

        fig.text(0.002,1.0-(bi+0.5)/n_blk,
                 f'Block {bi+1}\n{bstart}–{bend}\nRMSE={b_rmse:.3f}°C\nBias={b_bias:+.3f}°C',
                 va='center',ha='left',fontsize=7.5,transform=fig.transFigure,
                 bbox=dict(boxstyle='round,pad=0.3',facecolor='#EEF2FF',
                           edgecolor='#99A',alpha=0.90))

    plt.suptitle(
        f"{SCRIPT_NAME}  |  {month_label}  {part_label}  |  Rolling 7-Day SST Forecast\n"
        f"({TARGET_LAT}°N, {TARGET_LON}°E)   ★=target   Monthly RMSE:{m_rmse:.4f}°C",
        fontsize=11,fontweight='bold',y=0.97)
    plt.savefig(out_path,dpi=150,bbox_inches='tight',facecolor='white')
    plt.close(); print(f"    Saved {out_path.name}")


for month_name,m_start,m_end in MONTHS_INFO:
    print(f"\n  {month_name}")
    blocks=collect_blocks(m_start,m_end)
    if not blocks: print("    No blocks."); continue
    print(f"    {len(blocks)} blocks")
    m_mask=month_mask.get(m_start.month,np.zeros(n_days,bool))
    m_rmse=float(np.sqrt(np.mean((avg_series[m_mask]-gt_series[m_mask])**2))) \
           if m_mask.sum() else 0.0
    month_tag=month_name.replace(' ','_').lower()
    h1=blocks[:2]; h2=blocks[2:]
    if h1:
        end_day_h1=dates[h1[-1][-1]].day
        draw_spatial_png(h1,month_name,f"Part 1  Days 1–{end_day_h1}",
                         m_rmse,OUTPUT_DIR/f"plot1_spatial_{month_tag}_part1.png")
    if h2:
        d0=dates[h2[0][0]].day; d1=dates[h2[-1][-1]].day
        draw_spatial_png(h2,month_name,f"Part 2  Days {d0}–{d1}",
                         m_rmse,OUTPUT_DIR/f"plot1_spatial_{month_tag}_part2.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2: 90-DAY TIME SERIES
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*55}\nPLOT 2: TIME SERIES\n{'='*55}")

fig=plt.figure(figsize=(26,16))
gs2=GridSpec(3,1,fig,height_ratios=[3.5,1.1,0.85],
             hspace=0.06,top=0.91,bottom=0.07,left=0.07,right=0.96)
ax_main=fig.add_subplot(gs2[0])
ax_err =fig.add_subplot(gs2[1],sharex=ax_main)
ax_rmse=fig.add_subplot(gs2[2],sharex=ax_main)

x=np.arange(1,n_days+1)
date_labels=[d.strftime('%b %d') for d in pred_dates]
jan_end=int(month_mask[1].sum()); feb_end=jan_end+int(month_mask[2].sum())

shade_cfg=[(0.5,jan_end+0.5,'#AED6F1','January'),
           (jan_end+0.5,feb_end+0.5,'#A9DFBF','February'),
           (feb_end+0.5,n_days+0.5,'#F9CBA0','March')]
for ax in [ax_main,ax_err,ax_rmse]:
    for x0,x1,col,_ in shade_cfg:
        ax.axvspan(x0,x1,alpha=0.18,color=col,zorder=0)

ax_main.axhline(clim_mean,color='grey',lw=1.0,ls='--',alpha=0.40,
                label=f'Clim mean ({clim_mean:.2f}°C)',zorder=1)
ax_main.fill_between(x,min_series_c,max_series_c,alpha=0.13,color='tomato',
                     zorder=2,label='Min–max spread')

half_range = np.clip((max_series_c - min_series_c)/2, 0.02, None)
err_bar_vals = np.where(std_series > 0.01, std_series, half_range)
ax_main.errorbar(x,avg_series,yerr=err_bar_vals,fmt='none',ecolor='#CC3333',
                 elinewidth=1.1,capsize=3.5,capthick=1.1,alpha=0.55,zorder=3,
                 label='±1σ / spread')
ax_main.plot(x,gt_series,color='royalblue',lw=2.5,zorder=6,
             label='Ground Truth',alpha=0.93)
ax_main.plot(x,avg_series,color='crimson',lw=2.1,zorder=7,alpha=0.90,
             label=f'Predicted avg  RMSE={rmse_90:.4f}°C  R²={r2_val:.4f}  R={r_val:.4f}')

all_v=np.concatenate([gt_series,avg_series,min_series_c,max_series_c])
set_quarter_yticks(ax_main,all_v.min(),all_v.max())
style_ax(ax_main)

ylims=ax_main.get_ylim()
for x0,x1,col,lbl in shade_cfg:
    ax_main.text((x0+x1)/2,ylims[1]-0.02*(ylims[1]-ylims[0]),lbl,
                 ha='center',va='top',fontsize=11,color='#444',alpha=0.85,fontweight='bold')

for m_num in [1,2,3]:
    mm=month_mask[m_num]
    if mm.sum():
        mr=float(np.sqrt(np.mean((avg_series[mm]-gt_series[mm])**2)))
        me=float(np.mean(avg_series[mm]-gt_series[mm]))
        ax_main.text(float(x[mm].mean()),ylims[0]+0.03*(ylims[1]-ylims[0]),
                     f'RMSE={mr:.3f}°C\nBias={me:+.3f}°C',
                     ha='center',va='bottom',fontsize=8,
                     color=list(m_cols_ts.values())[m_num-1],
                     bbox=dict(boxstyle='round,pad=0.25',facecolor='white',
                               alpha=0.82,edgecolor=list(m_cols_ts.values())[m_num-1]))

ax_main.set_title(
    f'{SCRIPT_NAME}  —  90-Day Rolling SST Forecast  ({TARGET_LAT}°N, {TARGET_LON}°E)\n'
    f'{PRED_START_DATE.date()} → {PRED_END_DATE.date()}  SEQ={SEQ_LEN} H={HORIZON}  '
    f'Adaptive sliding bias correction (window={ADAPTIVE_WINDOW}d)\n'
    f'RMSE={rmse_90:.4f}°C  MAE={mae_90:.4f}°C  R²={r2_val:.4f}  R={r_val:.4f}',
    fontsize=11,fontweight='bold',pad=6)
ax_main.set_ylabel('SST (°C)',fontsize=12)
ax_main.legend(fontsize=9.5,loc='upper left',framealpha=0.93,ncol=2)

tbox=(f'RMSE       = {rmse_90:.4f}°C\n'
      f'MAE        = {mae_90:.4f}°C\n'
      f'R²         = {r2_val:.4f}\n'
      f'R          = {r_val:.4f}\n'
      f'Slope      = {slope_pt:.4f}\n'
      f'Intercept  = {interc_pt:.4f}°C\n'
      f'Max|err|   = {np.max(np.abs(avg_series-gt_series)):.4f}°C\n'
      f'Avg σ/day  = {std_series.mean():.4f}°C\n'
      f'Adapt win  = {ADAPTIVE_WINDOW}d\n'
      f'Avg offset = {adaptive_offsets.mean():+.4f}°C\n'
      f'n days     = {n_days}')
ax_main.text(0.994,0.97,tbox,transform=ax_main.transAxes,fontsize=8,
             va='top',ha='right',family='monospace',
             bbox=dict(boxstyle='round',facecolor='#FFFBEA',alpha=0.93,edgecolor='#BBB'))
plt.setp(ax_main.get_xticklabels(),visible=False)

err=avg_series-gt_series
bcols=['#27AE60' if abs(e)<0.05 else '#F39C12' if abs(e)<0.10 else '#E74C3C' for e in err]
ax_err.bar(x,err,color=bcols,alpha=0.85,edgecolor='#333',lw=0.35,zorder=3)
ax_err.errorbar(x,err,yerr=err_bar_vals,fmt='none',ecolor='#444',
                elinewidth=0.7,capsize=1.8,capthick=0.8,alpha=0.40,zorder=4)
ax_err.axhline(0,color='black',lw=1.8,zorder=5)
for th,col in [(0.05,'#27AE60'),(0.10,'#F39C12')]:
    ax_err.axhline( th,color=col,lw=1.3,ls='--',alpha=0.85,zorder=2)
    ax_err.axhline(-th,color=col,lw=1.3,ls='--',alpha=0.85,zorder=2)
err_ylim=max(abs(err.max()),abs(err.min()))*1.15
ax_err.set_ylim(-err_ylim,err_ylim)
for m_num in [1,2,3]:
    mm=month_mask[m_num]
    if mm.sum():
        me=float(err[mm].mean())
        ax_err.text(float(x[mm].mean()),err_ylim*0.90,
                    f'{month_names[m_num]}\nmean {me:+.3f}°C',
                    ha='center',va='top',fontsize=7.5,
                    color=list(m_cols_ts.values())[m_num-1],fontweight='bold')
legend_patches=[mpatches.Patch(color='#27AE60',label='|err|<0.05°C  ✓ good'),
                mpatches.Patch(color='#F39C12',label='|err|<0.10°C  ~ ok'),
                mpatches.Patch(color='#E74C3C',label='|err|≥0.10°C  ✗ high')]
ax_err.legend(handles=legend_patches,fontsize=7.5,loc='lower right',framealpha=0.88,ncol=3)
ax_err.set_ylabel('Error (°C)',fontsize=10)
ax_err.set_title('Per-Day Signed Error  (colour = error magnitude tier)',fontsize=9)
style_ax(ax_err); plt.setp(ax_err.get_xticklabels(),visible=False)

w_roll=7
roll_rmse=np.array([
    np.sqrt(np.mean((avg_series[max(0,i-w_roll+1):i+1]-
                     gt_series[max(0,i-w_roll+1):i+1])**2))
    for i in range(n_days)])
for m_num in [1,2,3]:
    mm=month_mask[m_num]; mr=roll_rmse[mm].mean() if mm.sum() else 0
    ax_rmse.fill_between(x,0,roll_rmse,where=mm,
                         color=list(m_cols_ts.values())[m_num-1],alpha=0.55,
                         label=f"{month_names[m_num]} avg {mr:.3f}°C")
ax_rmse.plot(x,roll_rmse,'k-',lw=1.2,alpha=0.60)
ax_rmse.axhline(rmse_90,color='red',lw=1.1,ls=':',alpha=0.65,
                label=f'Overall {rmse_90:.3f}°C')
ax_rmse.set_ylabel('7-day\nRMSE (°C)',fontsize=9)
ax_rmse.set_title('Rolling 7-Day RMSE',fontsize=9)
ax_rmse.legend(fontsize=8,loc='upper right',framealpha=0.88,ncol=4)
style_ax(ax_rmse); ax_rmse.set_ylim(bottom=0)

tick_every=max(1,n_days//20)
tick_pos=x[::tick_every]; tick_lbl=date_labels[::tick_every]
ax_rmse.set_xticks(tick_pos)
ax_rmse.set_xticklabels(tick_lbl,rotation=40,ha='right',fontsize=8.5)
for ax in [ax_main,ax_err,ax_rmse]: ax.set_xlim(0.5,n_days+0.5)

plt.savefig(OUTPUT_DIR/"plot2_timeseries_90day.png",dpi=150,bbox_inches='tight')
plt.close(); print("  Saved plot2_timeseries_90day.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 3: CORRELATION
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*55}\nPLOT 3: CORRELATION\n{'='*55}")

all_pred_flat=[]; all_act_flat=[]
for d_abs in pred_days_abs:
    if d_abs<T:
        all_pred_flat.append(avg_spatial[d_abs].ravel())
        all_act_flat.append(data_full[d_abs,:H_orig,:W_orig].ravel())
all_pred_flat=np.concatenate(all_pred_flat)
all_act_flat =np.concatenate(all_act_flat)
n_pts=len(all_pred_flat)
rng=np.random.default_rng(42)
idx=rng.choice(n_pts,min(n_pts,50000),replace=False)
pp,aa=all_pred_flat[idx],all_act_flat[idx]
r_all=float(np.corrcoef(all_pred_flat,all_act_flat)[0,1])
rmse_all=float(np.sqrt(np.mean((all_pred_flat-all_act_flat)**2)))
sl_all,ic_all,_,_,_=stats.linregress(all_act_flat,all_pred_flat)

fig,(ax_L,ax_R)=plt.subplots(1,2,figsize=(22,10))
fig.subplots_adjust(wspace=0.25,left=0.07,right=0.97,top=0.87,bottom=0.10)

hb=ax_L.hexbin(aa,pp,gridsize=90,cmap='Blues',mincnt=1,linewidths=0.12,bins='log')
cb=fig.colorbar(hb,ax=ax_L,pad=0.012,shrink=0.88)
cb.set_label('log₁₀(count)',fontsize=10); cb.ax.tick_params(labelsize=8)
vL=(min(aa.min(),pp.min())-0.2,max(aa.max(),pp.max())+0.2)
ax_L.plot(vL,vL,'k--',lw=2.0,alpha=0.60,label='Perfect (y=x)')
xf=np.linspace(vL[0],vL[1],200)
ax_L.plot(xf,sl_all*xf+ic_all,'r-',lw=2.3,alpha=0.80,
          label=f'Regression  slope={sl_all:.3f}')
ax_L.fill_between(xf,xf-0.3,xf+0.3,alpha=0.08,color='green',label='±0.3°C band')
ax_L.set_xlim(vL); ax_L.set_ylim(vL); ax_L.set_aspect('equal')
ax_L.set_xlabel('Actual SST (°C)',fontsize=13)
ax_L.set_ylabel('Predicted SST (°C)',fontsize=13)
ax_L.set_title(f'All Grid Points × {HORIZON} Days  (Jan–Mar 2026)\n'
               f'Pearson R={r_all:.4f}   RMSE={rmse_all:.4f}°C',
               fontsize=13,fontweight='bold')
ax_L.legend(fontsize=10,loc='upper left'); style_ax(ax_L)
ax_L.text(0.04,0.96,
          f'R       = {r_all:.4f}\nR²      = {r_all**2:.4f}\n'
          f'RMSE    = {rmse_all:.4f}°C\nSlope   = {sl_all:.4f}\n'
          f'Interc  = {ic_all:.4f}°C\nN pts   = {n_pts:,}',
          transform=ax_L.transAxes,fontsize=9.5,va='top',ha='left',family='monospace',
          bbox=dict(boxstyle='round',facecolor='#E8F4FD',alpha=0.93,edgecolor='#7BB'))

m_col_r={1:'#2471A3',2:'#1E8449',3:'#C0392B'}; handles_r=[]
for m_num in [1,2,3]:
    mm=month_mask[m_num]
    if mm.sum()==0: continue
    ax_R.scatter(gt_series[mm],avg_series[mm],c=m_col_r[m_num],s=70,
                 alpha=0.82,edgecolors='white',lw=0.5,zorder=5)
    handles_r.append(mpatches.Patch(color=m_col_r[m_num],
                     label=f"{month_names[m_num]} 2026  n={mm.sum()}"))
ax_R.errorbar(gt_series,avg_series,yerr=std_series,fmt='none',
              ecolor='#888',elinewidth=1.0,capsize=3.0,alpha=0.45,zorder=4)
vR=(min(gt_series.min(),avg_series.min())-0.25,
    max(gt_series.max(),avg_series.max())+0.25)
ax_R.plot(vR,vR,'k--',lw=2.0,alpha=0.58)
xfr=np.linspace(vR[0],vR[1],200)
ax_R.plot(xfr,slope_pt*xfr+interc_pt,'r-',lw=2.3,alpha=0.80,
          label=f'Regression  slope={slope_pt:.3f}')
ax_R.fill_between(xfr,xfr-0.3,xfr+0.3,alpha=0.08,color='green',label='±0.3°C band')
ax_R.set_xlim(vR); ax_R.set_ylim(vR); ax_R.set_aspect('equal')
ax_R.set_xlabel('Ground Truth SST (°C)',fontsize=13)
ax_R.set_ylabel('Predicted SST (°C)',fontsize=13)
ax_R.set_title(f'Target Pixel ({TARGET_LAT}°N, {TARGET_LON}°E)\n'
               f'R={r_val:.4f}  R²={r2_val:.4f}  RMSE={rmse_90:.4f}°C',
               fontsize=13,fontweight='bold')
handles_r+=[mpatches.Patch(color='none'),
            mpatches.Patch(color='k',label='Perfect (y=x)'),
            mpatches.Patch(color='r',label=f'Reg slope={slope_pt:.3f}'),
            mpatches.Patch(color='green',alpha=0.5,label='±0.3°C band')]
ax_R.legend(handles=handles_r,fontsize=9.5,loc='upper left',framealpha=0.92)
style_ax(ax_R)
ax_R.text(0.97,0.04,
          f'R       = {r_val:.4f}\nR²      = {r2_val:.4f}\n'
          f'RMSE    = {rmse_90:.4f}°C\nMAE     = {mae_90:.4f}°C\n'
          f'Slope   = {slope_pt:.4f}\nInterc  = {interc_pt:.4f}°C\nn days  = {n_days}',
          transform=ax_R.transAxes,fontsize=9.5,va='bottom',ha='right',family='monospace',
          bbox=dict(boxstyle='round',facecolor='#FFF8DC',alpha=0.93,edgecolor='#BBB'))
plt.suptitle(f'{SCRIPT_NAME}  —  Spatial & Point Correlation  |  '
             f'{PRED_START_DATE.date()} → {PRED_END_DATE.date()}  |  90 days',
             fontsize=14,fontweight='bold')
plt.savefig(OUTPUT_DIR/"plot3_correlation_scatter.png",dpi=150,bbox_inches='tight')
plt.close(); print("  Saved plot3_correlation_scatter.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 4: LOSS CONVERGENCE
# ═══════════════════════════════════════════════════════════════════════════════
loss_csv=OUTPUT_DIR/"loss_history.csv"
if loss_csv.exists():
    lh=pd.read_csv(loss_csv)
    fig,ax=plt.subplots(figsize=(10,6))
    ep_x=np.arange(1,len(lh)+1)
    ax.plot(ep_x,lh['train'],'b-o',lw=2,ms=5,label='Train')
    ax.plot(ep_x,lh['val'],'r-s',lw=2,ms=5,label='Val')
    ax.set_xlabel('Epoch',fontsize=12); ax.set_ylabel('Huber Loss',fontsize=12)
    ax.set_title(f'Loss Convergence  SEQ={SEQ_LEN} H={HORIZON} HIDDEN={HIDDEN_DIM}',
                 fontweight='bold',fontsize=11)
    ax.legend(fontsize=10); style_ax(ax)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"plot4_loss_convergence.png",dpi=150,bbox_inches='tight')
    plt.close(); print("  Saved plot4_loss_convergence.png")

# ═══════════════════════════════════════════════════════════════════════════════
# METRICS CSV + MONTHLY SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
pd.DataFrame({
    'date':[d.strftime('%Y-%m-%d') for d in pred_dates],
    'ground_truth':gt_series,'predicted_avg':avg_series,
    'predicted_std':std_series,'predicted_min':min_series_c,'predicted_max':max_series_c,
    'error':avg_series-gt_series,'abs_error':np.abs(avg_series-gt_series),
    'n_overlaps':n_overlaps,'adaptive_offset':adaptive_offsets,
}).to_csv(OUTPUT_DIR/"rolling_predictions.csv",index=False)

monthly_rows=[]
for m_num,m_lbl in month_names.items():
    mm=month_mask[m_num]
    if mm.sum():
        r_m=float(np.corrcoef(avg_series[mm],gt_series[mm])[0,1])
        monthly_rows.append({
            'month':f'{m_lbl} 2026','days':int(mm.sum()),
            'rmse':float(np.sqrt(np.mean((avg_series[mm]-gt_series[mm])**2))),
            'mae':float(np.mean(np.abs(avg_series[mm]-gt_series[mm]))),
            'r':r_m,'r2':r_m**2,
            'mean_adaptive_offset_C':float(adaptive_offsets[mm].mean())})
pd.DataFrame(monthly_rows).to_csv(OUTPUT_DIR/"monthly_summary.csv",index=False)

print(f"\n{'='*60}\nFINAL REPORT  {SCRIPT_NAME}")
print(f"  RMSE={rmse_90:.4f}°C  MAE={mae_90:.4f}°C  R²={r2_val:.4f}  R={r_val:.4f}")
print(f"  Slope={slope_pt:.4f}  Intercept={interc_pt:.4f}°C")
for ms in monthly_rows:
    print(f"  {ms['month']}: RMSE={ms['rmse']:.4f}°C  R²={ms['r2']:.4f}  "
          f"offset={ms['mean_adaptive_offset_C']:+.4f}°C")

zip_path=OUTPUT_DIR.parent/f"{OUTPUT_DIR.name}.zip"
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as zf:
    for f in OUTPUT_DIR.iterdir():
        if f.suffix!='.zip': zf.write(f,f.name)
print(f"\n  ZIP:{zip_path}")
try:
    from google.colab import files; files.download(str(zip_path)); print("  Download triggered.")
except Exception: print(f"  -> Manual download:{zip_path}")