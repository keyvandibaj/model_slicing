# ad-slicing.py  -- Hierarchical assigner + per-cluster VAE + Influx stream
# ASCII-only prints for safe terminals. Tested with Python 3.8+.

import os, sys, time, logging, traceback
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import joblib
import pickle
import torch
import torch.nn as nn
from influxdb import InfluxDBClient

# =========================
# Config
# =========================
QUERY_INTERVAL = 0.001  # seconds between polls
MODELS_DIR = Path("mosels_dir")  # folder containing hierarchical_assigner*.pkl and vae_* files
DEBUG = True  # set False to reduce stdout logs

# =========================
# Logging
# =========================
logging.basicConfig(
    filename='anomalies.log',
    level=logging.WARNING,
    format='%(asctime)s - %(message)s'
)

# =========================
# Unicode-safe stdout (optional)
# =========================
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# =======================================================
# Stub importer for unknown modules referenced in pickles
# e.g., DRB.* that may exist in legacy environments
# =======================================================
import types, importlib.abc, importlib.util

class _StubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        mod = types.ModuleType(spec.name)
        def __getattr__(name):
            # tolerant stub class: accepts any ctor args and state
            cls = type(name, (object,), {})
            def __new__(c, *a, **k): return object.__new__(c)
            def __init__(self, *a, **k): pass
            def __setstate__(self, state):
                try:
                    if isinstance(state, dict):
                        self.__dict__.update(state)
                    else:
                        self.__dict__['_state'] = state
                except Exception:
                    pass
            cls.__new__ = staticmethod(__new__)
            cls.__init__ = __init__
            cls.__setstate__ = __setstate__
            return cls
        mod.__getattr__ = __getattr__  # PEP 562
        return mod
    def exec_module(self, module):
        return

class _StubFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        # Stub DRB and any submodules DRB.*
        if fullname == "DRB" or fullname.startswith("DRB."):
            return importlib.util.spec_from_loader(fullname, _StubLoader())
        return None

if not any(isinstance(f, _StubFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _StubFinder())

# =========================
# VAE (as provided)
# =========================
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

# =========================
# Assigner loader (clean-first)
# =========================
NEEDED_KEYS = {"selected_columns", "scaler", "centroids", "invs", "distance_metric"}

def _clean_assigner_dict(b: dict) -> dict:
    """Keep only required keys and normalize types."""
    clean = {}
    clean["selected_columns"] = list(b["selected_columns"])
    clean["scaler"] = b["scaler"]
    clean["centroids"] = np.asarray(b["centroids"])
    clean["invs"] = [np.asarray(m) for m in b["invs"]]
    dm = b.get("distance_metric", "euclidean")
    clean["distance_metric"] = str(dm)
    return clean

def load_assigner_auto(models_dir: Path) -> dict:
    """Load hierarchical assigner:
       1) if clean exists -> load it
       2) else load legacy (with stubbed imports), then write clean and return it
    """
    models_dir = Path(models_dir)
    clean_path = models_dir / "hierarchical_assigner_clean.pkl"
    raw_path   = models_dir / "hierarchical_assigner.pkl"

    if clean_path.exists():
        b = joblib.load(clean_path)
        missing = NEEDED_KEYS - set(b.keys())
        if missing:
            raise ValueError("[assigner_clean] missing keys: {}".format(missing))
        return b

    # legacy path
    try:
        b_raw = joblib.load(raw_path)
    except Exception:
        with open(raw_path, "rb") as f:
            b_raw = pickle.load(f)

    b = _clean_assigner_dict(b_raw)
    try:
        joblib.dump(b, clean_path, compress=3)
        print("[ASSIGNER] wrote clean file:", clean_path)
    except Exception as e:
        print("[ASSIGNER][warn] could not write clean file:", e)
    return b

# =========================
# Utils
# =========================
def _align_columns_for_scaler(df: pd.DataFrame, cols_expected: list, scaler) -> pd.DataFrame:
    """Align df columns to scaler expected order, create missing, coerce to numeric, fill NaN with median."""
    if hasattr(scaler, "feature_names_in_"):
        expected = list(scaler.feature_names_in_)
    else:
        expected = list(cols_expected)

    for c in expected:
        if c not in df.columns:
            df[c] = np.nan

    X = df[expected].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    for c in expected:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X.fillna(X.median(numeric_only=True), inplace=True)
    return X

def load_cluster_vaes(models_dir: Path, K: int) -> Dict[int, Dict[str, Any]]:
    """Load per-cluster VAE packs: (vae_model_class{k}.pth, vae_bundle_class{k}.pkl)."""
    vaes = {}
    for k in range(K):
        pth = models_dir / f"vae_model_class{k}.pth"
        pkl = models_dir / f"vae_bundle_class{k}.pkl"
        if not pth.exists() or not pkl.exists():
            logging.warning("[INIT] Missing VAE files for cluster {}: {} / {}".format(k, pth.name, pkl.name))
            continue
        bundle = joblib.load(pkl)
        feats, scaler, thr = bundle["features"], bundle["scaler"], float(bundle["threshold"])
        vae = VAE(input_dim=len(feats))
        vae.load_state_dict(torch.load(pth, map_location="cpu"))
        vae.eval()
        vaes[k] = {"vae": vae, "features": feats, "scaler": scaler, "threshold": thr}
    if not vaes:
        raise RuntimeError("No per-cluster VAE models found.")
    return vaes

# =========================
# Cluster assignment for one row
# =========================
def assign_cluster_for_row(row_dict: Dict[str, Any], assigner: Dict[str, Any]) -> int:
    cols = assigner["selected_columns"]
    scaler = assigner["scaler"]
    C = assigner["centroids"]
    invs = assigner["invs"]
    metric = assigner["distance_metric"]

    df_row = pd.DataFrame([row_dict])
    Xn = _align_columns_for_scaler(df_row, cols_expected=cols, scaler=scaler)
    Xs = scaler.transform(Xn)  # (1, D)

    if metric == "euclidean":
        dists = np.linalg.norm(Xs - C, axis=1)
        return int(np.argmin(dists))

    elif metric == "mahalanobis":
        K = C.shape[0]
        d = np.zeros(K)
        for k in range(K):
            diff = Xs[0] - C[k]
            d[k] = float(diff @ invs[k] @ diff)
        return int(np.argmin(d))

    else:
        raise ValueError("distance_metric must be 'euclidean' or 'mahalanobis'.")

# =========================
# VAE inference for one row
# =========================
def vae_anomaly_for_row(row_dict: Dict[str, Any], vae_pack: Dict[str, Any]) -> Tuple[float, int]:
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

# =========================
# InfluxDB streaming (3 buckets)
# =========================
def query_influxdb():
    """
    Streams joined rows from:
      - cu_cp_bucket: base + neighbors
      - du_bucket: "drb_uethp_dl_ueid", rru_prb_used_dl, tb_err_total_nbr_dl_1
      - cu_up_bucket: "drb_pdcp_sdu_volume_dl_filter_ueid_tx_bytes",
                      "tot_pdcp_sdu_nbr_dl_ueid_tx_dl_packets",
                      "drb_pdcp_sdu_delay_dl_ueid_pdcp_latency"
    Join key: (ue_imsi_complete, time)
    """
    client = InfluxDBClient(host='localhost', port=8086, database='ns3_metrics')
    last_time_seen = "1970-01-01T00:00:00.000000Z"

    while True:
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

            if t > last_time_seen:
                last_time_seen = t

            du_point = du_lookup.get((ueid, t))
            cuup_point = cuup_lookup.get((ueid, t))
            if not du_point or not cuup_point:
                continue

            # Build row with keys matching your models/bundles
            row = {
                'ueid': ueid,
                'time': t,

                # CU-CP
                'numActiveUes': cu_point.get('num_active_ues'),
                'L3 serving SINR': cu_point.get('l3_serving_sinr'),
                'L3 serving SINR 3gpp': cu_point.get('l3_serving_sinr_3gpp'),
                'neighbor_id_1': cu_point.get('l3_neigh_id_1_cellid'),
                'neighbor_id_2': cu_point.get('l3_neigh_id_2_cellid'),
                'neighbor_id_3': cu_point.get('l3_neigh_id_3_cellid'),
                'neighbor_sinr_1': cu_point.get('l3_neigh_sinr_1'),
                'neighbor_sinr_2': cu_point.get('l3_neigh_sinr_2'),
                'neighbor_sinr_3': cu_point.get('l3_neigh_sinr_3'),

                # DU
                'RRU.PrbUsedDl': du_point.get('rru_prb_used_dl'),
                'TB.ErrTotalNbrDl.1': du_point.get('tb_err_total_nbr_dl_1'),
                'DRB.UEThpDl.UEID': du_point.get('drb_uethp_dl_ueid'),

                # CU-UP
                'DRB.PdcpSduVolumeDl_Filter.UEID(txBytes)': cuup_point.get('drb_pdcp_sdu_volume_dl_filter_ueid_tx_bytes'),
                'Tot.PdcpSduNbrDl.UEID(txDlPackets)': cuup_point.get('tot_pdcp_sdu_nbr_dl_ueid_tx_dl_packets'),
                'DRB.PdcpSduDelayDl.UEID(pdcpLatency)': cuup_point.get('drb_pdcp_sdu_delay_dl_ueid_pdcp_latency'),
            }

            # Soft-cast to float when possible
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
                    try:
                        row[k] = float(v)
                    except Exception:
                        pass

            yield row

        time.sleep(QUERY_INTERVAL)

# =========================
# Dry-run: quick self-test without Influx
# =========================
def dry_run_once():
    """Run one synthetic example through the pipeline to verify artifacts."""
    try:
        assigner = load_assigner_auto(MODELS_DIR)
        K = int(assigner["centroids"].shape[0])
        vaes = load_cluster_vaes(MODELS_DIR, K)
        print("[DRY] clusters:", K, "| VAE packs:", sorted(list(vaes.keys())))

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
        print("[DRY] rec_err={:.6f}  thr={:.6f}  pred={}".format(rec_err, vaes[k]['threshold'], y_hat))
    except Exception as e:
        print("[DRY][ERROR]", e)
        traceback.print_exc()

# =========================
# Main
# =========================
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

        print("[INIT] Loading assigner...")
        assigner = load_assigner_auto(MODELS_DIR)
        K = int(assigner["centroids"].shape[0])
        print("[INIT] Clusters (K):", K)
        print("[CHECK] keys:", sorted(list(assigner.keys())))
        print("[CHECK] centroids shape:", getattr(assigner["centroids"], "shape", None))
        print("[CHECK] invs len:", len(assigner["invs"]))
        print("[CHECK] distance_metric:", assigner["distance_metric"])

        print("[INIT] Loading VAEs...")
        vaes = load_cluster_vaes(MODELS_DIR, K)
        print("[INIT] Loaded VAE packs:", sorted(list(vaes.keys())))

        print("[INIT] Running DRY-RUN once...")
        dry_run_once()
        print("[INIT] DRY-RUN done.")

        print("\n[RUN] Starting stream (every {}s)...\n".format(QUERY_INTERVAL))
        print("Time                          | UEID   | Cluster |  rec_err  |    thr   | ServingSINR | Neigh( id:sinr, ... )")
        print("-" * 130)

        for row in query_influxdb():
            try:
                if DEBUG:
                    print("[ROW] keys:", list(row.keys())[:8], "... total:", len(row))

                k = assign_cluster_for_row(row, assigner)
                if k not in vaes:
                    if DEBUG:
                        print("[WARN] No VAE pack for cluster {}. Skipping row.".format(k))
                    continue

                rec_err, y_hat = vae_anomaly_for_row(row, vaes[k])
                thr = vaes[k]['threshold']

                serving_sinr = row.get('L3 serving SINR', None)
                s_sinr_str = "{:.2f}".format(serving_sinr) if isinstance(serving_sinr, (int, float)) else "N/A"
                neigh_str = "{}:{}, {}:{}, {}:{}".format(
                    row.get('neighbor_id_1'), row.get('neighbor_sinr_1'),
                    row.get('neighbor_id_2'), row.get('neighbor_sinr_2'),
                    row.get('neighbor_id_3'), row.get('neighbor_sinr_3')
                )

                print("{} | {:6d} | {:^7d} | {:9.4f} | {:9.4f} | {:>11} | {}".format(
                    row['time'], int(row['ueid']), k, rec_err, thr, s_sinr_str, neigh_str))

                if y_hat == 1:
                    msg = ("UE {} anomalous at {} (cluster={}, rec_err={:.4f}, thr={:.4f}); "
                           "ServingSINR={}; Neigh={}; RowKeys={}").format(
                        int(row['ueid']), row['time'], k, rec_err, thr, s_sinr_str, neigh_str, list(row.keys()))
                    logging.warning(msg)
                    print("\n[ALERT] {}\n".format(msg))

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
        print("\nShutting down...")
    except Exception as outer_e:
        print("[CRITICAL] Fatal error:", outer_e)
        traceback.print_exc()
        try:
            print("[CRITICAL] MODELS_DIR contents:", os.listdir(MODELS_DIR))
        except Exception:
            pass

# =========================
# Entry
# =========================
if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    print("[BOOT] __main__ guard active. Calling main() ...")
    main()
