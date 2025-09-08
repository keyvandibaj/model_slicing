"""
Real-Time Anomaly Detection with Hierarchical Assigner + Per-Cluster VAE
- Streams from InfluxDB
- Assigns cluster using hierarchical_assigner.pkl
- Runs the VAE of that cluster with its own scaler & threshold
- Logs anomalies with rich context
"""

import os, time, logging
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from influxdb import InfluxDBClient
import sys, traceback
DEBUG = True
print("[BOOT] ad-slicing.py starting…")
print("[BOOT] Python:", sys.version)

# --- add this at the very top with imports ---
try:
    import pickle5 as pickle
except ImportError:
    import pickle
def dry_run_once():
    """بدون اتصال به Influx یک رکورد آزمایشی می‌سازد و پایپ‌لاین را تست می‌کند."""
    try:
        assigner = load_assigner(MODELS_DIR / "hierarchical_assigner.pkl")
        K = int(assigner["centroids"].shape[0])
        vaes = load_cluster_vaes(MODELS_DIR, K)
        print("[DRY] clusters:", K, "| VAE packs:", sorted(list(vaes.keys())))

        # یک ردیف آزمایشی حداقلی (کلیدها باید با مدل/باندل تو سازگار باشند)
        row = {
            'ueid': 123456,
            'time': "1970-01-01T00:00:01Z",
            'numActiveUes': 1.0,
            'L3 serving SINR': 10.0,
            'L3 serving SINR 3gpp': 8.0,
            'neighbor_id_1': None, 'neighbor_id_2': None, 'neighbor_id_3': None,
            'neighbor_sinr_1': None, 'neighbor_sinr_2': None, 'neighbor_sinr_3': None,
            'RRU.PrbUsedDl': 0.1,
            'TB.ErrTotalNbrDl.1': 0.0,
            # چهار فیچر PDCP/UEThp که گفتی:
            'DRB.UEThpDl.UEID': 50.0,
            'DRB.PdcpSduVolumeDl_Filter.UEID(txBytes)': 1000.0,
            'Tot.PdcpSduNbrDl.UEID(txDlPackets)': 10.0,
            'DRB.PdcpSduDelayDl.UEID(pdcpLatency)': 5.0,
        }

        k = assign_cluster_for_row(row, assigner)
        print("[DRY] assigned cluster:", k)
        if k not in vaes:
            print("[DRY][WARN] No VAE pack for cluster", k)
            return
        rec_err, y_hat = vae_anomaly_for_row(row, vaes[k])
        print(f"[DRY] rec_err={rec_err:.6f}  thr={vaes[k]['threshold']:.6f}  pred={y_hat}")
    except Exception as e:
        print("[DRY][ERROR]", e)
        traceback.print_exc()

# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(
    filename='anomalies.log',
    level=logging.WARNING,
    format='%(asctime)s - %(message)s'
)

# -------------------------------------------------
# Config
# -------------------------------------------------
QUERY_INTERVAL = 0.001  # seconds
MODELS_DIR = Path("mosels_dir")  # پوشه‌ی آرتیفکت‌ها؛ در صورت نیاز عوض کن
# اگر بخشی از فیچرها در Influx نیستند، کد به‌صورت امن NaN می‌سازد و با میانه پر می‌کند

# -------------------------------------------------
# VAE (مطابق معماری شما)
# -------------------------------------------------
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=4):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU()
        )
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def _align_columns_for_scaler(df: pd.DataFrame, cols_expected: list, scaler) -> pd.DataFrame:
    """
    هم‌ترازسازی نام/ترتیب ستون‌ها با اسکالرِ فیت‌شده.
    - ستون‌های گم‌شده ساخته می‌شوند و با میانه پر می‌گردند.
    - ستون‌های اضافه حذف می‌شوند.
    - انواع غیرعددی به عدد تبدیل می‌شوند.
    """
    if hasattr(scaler, "feature_names_in_"):
        expected = list(scaler.feature_names_in_)
    else:
        expected = list(cols_expected)

    # ساخت ستون‌های گم‌شده
    for c in expected:
        if c not in df.columns:
            df[c] = np.nan

    # فقط ستون‌های مورد انتظار
    X = df[expected].copy()

    # پاک‌سازی
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    for c in expected:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X.fillna(X.median(numeric_only=True), inplace=True)
    return X

# -------------------------------------------------
# Load artifacts
# -------------------------------------------------
def load_assigner(assigner_path: Path) -> Dict[str, Any]:
    b = safe_load_pkl(assigner_path)
    required = {"selected_columns", "scaler", "centroids", "invs", "distance_metric"}
    missing = required - set(b.keys())
    if missing:
        raise ValueError(f"Assigner file missing keys: {missing}")
    return b


def load_cluster_vaes(models_dir: Path, K: int) -> Dict[int, Dict[str, Any]]:
    """
    برای هر خوشه k فایل‌های:
      vae_model_class{k}.pth
      vae_bundle_class{k}.pkl  (features, scaler, threshold)
    را لود می‌کند. اگر فایلی نبود، آن خوشه غیرفعال می‌ماند.
    """
    vaes = {}
    for k in range(K):
        pth = models_dir / f"vae_model_class{k}.pth"
        pkl = models_dir / f"vae_bundle_class{k}.pkl"
        if not pth.exists() or not pkl.exists():
            logging.warning(f"[INIT] Missing VAE files for cluster {k}: {pth.name} / {pkl.name}")
            continue
        bundle = safe_load_pkl(pkl)
        #bundle = joblib.load(pkl)
        feats, scaler, thr = bundle["features"], bundle["scaler"], float(bundle["threshold"])
        vae = VAE(input_dim=len(feats))
        vae.load_state_dict(torch.load(pth, map_location="cpu", pickle_module=pickle))
        #vae.load_state_dict(torch.load(pth, map_location="cpu"))
        vae.eval()
        vaes[k] = {"vae": vae, "features": feats, "scaler": scaler, "threshold": thr}
    if not vaes:
        raise RuntimeError("No per-cluster VAE models found.")
    return vaes

# -------------------------------------------------
# Clustering (assign cluster for one row)
# -------------------------------------------------
def assign_cluster_for_row(row_dict: Dict[str, Any], assigner: Dict[str, Any]) -> int:
    cols = assigner["selected_columns"]
    scaler = assigner["scaler"]
    C = assigner["centroids"]
    invs = assigner["invs"]
    metric = assigner["distance_metric"]

    df_row = pd.DataFrame([row_dict])
    Xn = _align_columns_for_scaler(df_row, cols_expected=cols, scaler=scaler)
    Xs = scaler.transform(Xn)  # shape (1, D)

    if metric == "euclidean":
        dists = np.linalg.norm(Xs - C, axis=1)             # (K,)
        label = int(np.argmin(dists))
    elif metric == "mahalanobis":
        K = C.shape[0]
        d = np.zeros(K)
        for k in range(K):
            diff = Xs[0] - C[k]
            d[k] = float(diff @ invs[k] @ diff)
        label = int(np.argmin(d))
    else:
        raise ValueError("distance_metric must be 'euclidean' or 'mahalanobis'.")
    return label

# -------------------------------------------------
# VAE inference for one row (given its cluster)
# -------------------------------------------------
def vae_anomaly_for_row(row_dict: Dict[str, Any], vae_pack: Dict[str, Any]) -> Tuple[float, int]:
    """
    بازگشت:
      rec_err, predicted_label (0/1)
    """
    vae = vae_pack["vae"]
    feats = vae_pack["features"]
    scaler = vae_pack["scaler"]
    thr = vae_pack["threshold"]

    df_row = pd.DataFrame([row_dict])
    X = _align_columns_for_scaler(df_row, cols_expected=feats, scaler=scaler)
    Xs = scaler.transform(X)
    x = torch.as_tensor(Xs, dtype=torch.float32)

    with torch.no_grad():
        recon, _, _ = vae(x)
        rec_err = float(((recon - x)**2).mean().item())
    y_hat = int(rec_err > thr)
    return rec_err, y_hat
def safe_load_pkl(path):
    """
    Loads a pickle/joblib file even if it references unknown modules/classes like 'DRB'.
    We remap any class from module 'DRB' (or submodules) to lightweight stubs so unpickling succeeds.
    Then we normalize known fields to plain Python types.
    """
    # 1) ابتدا با joblib/pickle تلاش معمولی
    try:
        return joblib.load(path)
    except Exception as e1:
        # اگر مشکل از نبودن ماژول DRB است، می‌ریم سراغ Unpickler سفارشی
        msg = str(e1)
        if "No module named 'DRB'" not in msg and "No module named \\x27DRB\\x27" not in msg:
            # شاید pickle5 بخواد:
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        # ادامه می‌دیم با DRBUnpickler

    # 2) Unpickler سفارشی: هر کلاس از module == 'DRB' یا prefix 'DRB.' را به یک کلاس ساختگی نگاشت می‌کند
    with open(path, "rb") as f:

        class DRBUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # هر چیزی از ماژول DRB یا زیربخش‌هاش، stub می‌شود
                if module == "DRB" or module.startswith("DRB."):
                    # کلاس ساختگی داینامیک با همان نام تا ساخت نمونه شکست نخورد
                    return type(name, (object,), {})
                # در غیر اینصورت رفتار عادی
                return super().find_class(module, name)

        obj = DRBUnpickler(f).load()

    # 3) نرمال‌سازی فیلدهای شناخته‌شده برای assigner
    #    (اگر distance_metric به صورت Enum/آبجکت ذخیره شده بود -> str)
    if isinstance(obj, dict):
        if "distance_metric" in obj and not isinstance(obj["distance_metric"], str):
            try:
                obj["distance_metric"] = str(obj["distance_metric"])
            except Exception:
                obj["distance_metric"] = "euclidean"  # پیش‌فرض امن
    return obj


# -------------------------------------------------
# InfluxDB stream (بدون تغییر اساسی در اسکلت)
# -------------------------------------------------
def query_influxdb():
    """
    Stream از سه bucket: cu_cp_bucket, du_bucket, cu_up_bucket
    - فیلدهای با کاراکتر خاص با "..." کوت می‌شوند.
    - join روی (ue_imsi_complete, time)
    """
    client = InfluxDBClient(host='localhost', port=8086, database='ns3_metrics')
    last_time_seen = "1970-01-01T00:00:00.000000Z"

    while True:
        # --- CU-CP: پایه و زمان/UE ---
        cu_query = f"""
            SELECT
                num_active_ues,
                l3_serving_sinr,
                l3_serving_sinr_3gpp,
                ue_imsi_complete,
                time,
                l3_neigh_id_1_cellid,
                l3_neigh_id_2_cellid,
                l3_neigh_id_3_cellid,
                l3_neigh_sinr_1,
                l3_neigh_sinr_2,
                l3_neigh_sinr_3
            FROM cu_cp_bucket
            WHERE time > '{last_time_seen}'
            ORDER BY time ASC
        """
        cu_points = list(client.query(cu_query).get_points())

        if not cu_points:
            time.sleep(QUERY_INTERVAL)
            continue

        # --- DU: فقط نرخ DL UEThp (اسم با دابل‌کوت) ---
        du_query = """
            SELECT
                "drb_uethp_dl_ueid",
                rru_prb_used_dl,
                tb_err_total_nbr_dl_1,
                ue_imsi_complete,
                time
            FROM du_bucket
            ORDER BY time ASC
        """
        du_points = list(client.query(du_query).get_points())
        du_lookup = {(p['ue_imsi_complete'], p['time']): p for p in du_points}

        # --- CU-UP: سه فیچر PDCP (همه با دابل‌کوت) ---
        cuup_query = """
            SELECT
                "drb_pdcp_sdu_volume_dl_filter_ueid_tx_bytes",
                "tot_pdcp_sdu_nbr_dl_ueid_tx_dl_packets",
                "drb_pdcp_sdu_delay_dl_ueid_pdcp_latency",
                ue_imsi_complete,
                time
            FROM cu_up_bucket
            ORDER BY time ASC
        """
        cuup_points = list(client.query(cuup_query).get_points())
        cuup_lookup = {(p['ue_imsi_complete'], p['time']): p for p in cuup_points}

        for cu_point in cu_points:
            ueid = cu_point['ue_imsi_complete']
            t = cu_point['time']

            # به‌روزرسانی نشانگر زمان با حاشیهٔ امن
            if t > last_time_seen:
                last_time_seen = t

            du_point = du_lookup.get((ueid, t))
            cuup_point = cuup_lookup.get((ueid, t))
            if not du_point or not cuup_point:
                # اگر یکی از باکت‌ها رکورد هم‌زمان نداشت، رد کن
                continue

            # ردیف کامل با کلیدهایی دقیقاً مطابق باندل/مدل
            row = {
                # شناسه‌ها/زمان
                'ueid': ueid,
                'time': t,

                # پایه از CU-CP
                'numActiveUes': cu_point.get('num_active_ues'),
                'L3 serving SINR': cu_point.get('l3_serving_sinr'),
                'L3 serving SINR 3gpp': cu_point.get('l3_serving_sinr_3gpp'),
                'neighbor_id_1': cu_point.get('l3_neigh_id_1_cellid'),
                'neighbor_id_2': cu_point.get('l3_neigh_id_2_cellid'),
                'neighbor_id_3': cu_point.get('l3_neigh_id_3_cellid'),
                'neighbor_sinr_1': cu_point.get('l3_neigh_sinr_1'),
                'neighbor_sinr_2': cu_point.get('l3_neigh_sinr_2'),
                'neighbor_sinr_3': cu_point.get('l3_neigh_sinr_3'),

                # از DU
                'RRU.PrbUsedDl': du_point.get('rru_prb_used_dl'),
                'TB.ErrTotalNbrDl.1': du_point.get('tb_err_total_nbr_dl_1'),
                'DRB.UEThpDl.UEID': du_point.get('drb_uethp_dl_ueid'),

                # از CU-UP (اسم‌ها را دقیق می‌گذاریم)
                'DRB.PdcpSduVolumeDl_Filter.UEID(txBytes)': cuup_point.get('drb_pdcp_sdu_volume_dl_filter_ueid_tx_bytes'),
                'Tot.PdcpSduNbrDl.UEID(txDlPackets)': cuup_point.get('tot_pdcp_sdu_nbr_dl_ueid_tx_dl_packets'),
                'DRB.PdcpSduDelayDl.UEID(pdcpLatency)': cuup_point.get('drb_pdcp_sdu_delay_dl_ueid_pdcp_latency'),
            }

            # تبدیل نرم به float برای فیلدهای عددی (از خطای type جلوگیری می‌کند)
            for k in [
                'numActiveUes','L3 serving SINR','L3 serving SINR 3gpp',
                'RRU.PrbUsedDl','TB.ErrTotalNbrDl.1',
                'DRB.UEThpDl.UEID',
                'DRB.PdcpSduVolumeDl_Filter.UEID(txBytes)',
                'Tot.PdcpSduNbrDl.UEID(txDlPackets)',
                'DRB.PdcpSduDelayDl.UEID(pdcpLatency)'
            ]:
                v = row.get(k, None)
                if v is not None:
                    try: row[k] = float(v)
                    except: pass

            yield row

        time.sleep(QUERY_INTERVAL)

# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    try:
        print("[INIT] MODELS_DIR:", MODELS_DIR.resolve())
        if not MODELS_DIR.exists():
            print("[INIT][ERROR] MODELS_DIR not found!")
        else:
            try:
                print("[INIT] MODELS_DIR contents:", os.listdir(MODELS_DIR))
            except Exception as e:
                print("[INIT][WARN] listdir failed:", e)

        # 1) لود assigner
        print("[INIT] Loading assigner…")
        assigner_path = MODELS_DIR / "hierarchical_assigner.pkl"
        print("[INIT] Assigner path:", assigner_path)
        assigner = load_assigner(assigner_path)
        K = int(assigner["centroids"].shape[0])
        print("[INIT] Clusters (K):", K)

        # 2) لود VAEهای هر خوشه
        print("[INIT] Loading VAEs…")
        vaes = load_cluster_vaes(MODELS_DIR, K)
        print("[INIT] Loaded VAE packs:", sorted(list(vaes.keys())))

        # 3) یک Dry-run قبل از استریم
        print("[INIT] Running DRY-RUN once…")
        dry_run_once()
        print("[INIT] DRY-RUN done.")

        print(f"\n[RUN] Starting stream (every {QUERY_INTERVAL}s)…\n")
        print("Time                          | UEID   | Cluster | rec_err |    thr | ServingSINR | Neigh( id:sinr, ... )")
        print("-" * 130)

        # 4) استریم واقعی
        for row in query_influxdb():
            try:
                if DEBUG:
                    print("[ROW] keys:", list(row.keys())[:8], "… total:", len(row))

                k = assign_cluster_for_row(row, assigner)
                if k not in vaes:
                    if DEBUG:
                        print(f"[WARN] No VAE pack for cluster {k}. Skipping row.")
                    continue

                rec_err, y_hat = vae_anomaly_for_row(row, vaes[k])
                thr = vaes[k]['threshold']

                serving_sinr = row.get('L3 serving SINR', None)
                s_sinr_str = f"{serving_sinr:.2f}" if isinstance(serving_sinr, (int, float)) else "N/A"
                neigh_str = (
                    f"{row.get('neighbor_id_1')}:{row.get('neighbor_sinr_1')}, "
                    f"{row.get('neighbor_id_2')}:{row.get('neighbor_sinr_2')}, "
                    f"{row.get('neighbor_id_3')}:{row.get('neighbor_sinr_3')}"
                )

                print(f"{row['time']} | {int(row['ueid']):6d} | {k:^7d} | "
                      f"{rec_err:9.4f} | {thr:9.4f} | {s_sinr_str:>11} | {neigh_str}")

                if y_hat == 1:
                    msg = (f"UE {int(row['ueid'])} anomalous at {row['time']} "
                           f"(cluster={k}, rec_err={rec_err:.4f}, thr={thr:.4f}); "
                           f"ServingSINR={s_sinr_str}; Neigh={neigh_str}; "
                           f"RowKeys={list(row.keys())}")
                    logging.warning(msg)
                    print(f"\n[ALERT] {msg}\n")

                time.sleep(QUERY_INTERVAL)

            except Exception as inner_e:
                print("[ERROR] Iteration error:", inner_e)
                traceback.print_exc()
                try:
                    from pprint import pformat
                    print("[ERROR] Offending row dump:\n", pformat(row))
                except Exception:
                    pass
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nShutting down…")

    except Exception as outer_e:
        print("[CRITICAL] Fatal error:", outer_e)
        traceback.print_exc()
        try:
            print("[CRITICAL] MODELS_DIR contents:", os.listdir(MODELS_DIR))
        except Exception:
            pass
if __name__ == "__main__":
    # خروجی Unbuffered برای دیدن فوری پرینت‌ها
    try:
        import os
        os.environ["PYTHONUNBUFFERED"] = "1"
    except Exception:
        pass
    print("[BOOT] __main__ guard active. Calling main() …")
    main()
