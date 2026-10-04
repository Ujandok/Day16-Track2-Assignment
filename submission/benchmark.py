import json, os, time
import lightgbm as lgb, pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
B = os.path.expanduser("~/ml-benchmark")
t0 = time.perf_counter(); df = pd.read_csv(f"{B}/creditcard.csv"); load_t = time.perf_counter() - t0
X, y = df.drop(columns=["Class"]), df["Class"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.1, stratify=ytr, random_state=42)
m = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, random_state=42, reg_lambda=1.0, min_child_samples=50, verbose=-1)
t0 = time.perf_counter()
m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc", callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)])
train_t = time.perf_counter() - t0
p = m.predict_proba(Xte)[:, 1]; pred = (p >= 0.5).astype(int)
row = Xte.iloc[[0]]
for _ in range(20): m.predict_proba(row)
t0 = time.perf_counter()
for _ in range(200): m.predict_proba(row)
lat_ms = (time.perf_counter() - t0) / 200 * 1000
t0 = time.perf_counter(); m.predict_proba(Xte.iloc[:1000]); bt = time.perf_counter() - t0
r = {"rows": len(df), "cpu_count": os.cpu_count(), "load_time_s": round(load_t, 3), "train_time_s": round(train_t, 3),
     "best_iteration": int(m.best_iteration_ or 1000), "auc_roc": round(roc_auc_score(yte, p), 4),
     "accuracy": round(accuracy_score(yte, pred), 4), "f1_score": round(f1_score(yte, pred), 4),
     "precision": round(precision_score(yte, pred), 4), "recall": round(recall_score(yte, pred), 4),
     "latency_1row_ms": round(lat_ms, 3), "batch_1000rows_ms": round(bt * 1000, 3),
     "throughput_rows_per_s": round(1000 / bt, 1)}
for k, v in r.items(): print(f"{k:24s}: {v}")
json.dump(r, open(f"{B}/benchmark_result.json", "w"), indent=2)
print("Saved benchmark_result.json")
