import importlib.util
import json
import sys
import traceback
import types
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sample_code_submission.boosted_decision_tree import BoostedDecisionTree

if 'HiggsML' not in sys.modules:
    higgsml = types.ModuleType('HiggsML')
    higgsml.__path__ = []
    sys.modules['HiggsML'] = higgsml
if 'HiggsML.systematics' not in sys.modules:
    hs = types.ModuleType('HiggsML.systematics')
    hs.systematics = object()
    sys.modules['HiggsML.systematics'] = hs
if 'systematic_analysis' not in sys.modules:
    sa = types.ModuleType('systematic_analysis')
    sa.__path__ = []
    sys.modules['systematic_analysis'] = sa
if 'systematic_analysis.py' not in sys.modules:
    sapy = types.ModuleType('systematic_analysis.py')
    sys.modules['systematic_analysis.py'] = sapy

out_path = Path(r'c:/Users/mathi/Documents/ei/Alpaca-new/Higgs_collaborations/tmp_test_compute_mu_unbinned_result.json')

try:
    sa_path = Path(r'c:/Users/mathi/Documents/ei/Alpaca-new/Higgs_collaborations/sample_code_submission/statistical_analysis.py')
    spec = importlib.util.spec_from_file_location('stat_analysis', sa_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    parquet_path = Path(r'c:/Users/mathi/Documents/ei/Alpaca-new/Higgs_collaborations/blackSwan_data/blackSwan_data.parquet')
    df = pd.read_parquet(parquet_path)

    max_rows = 60000
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)

    labels = df['labels'].to_numpy(dtype=int)
    weights = df['weights'].to_numpy(dtype=float)
    feature_cols = [c for c in df.columns if c not in {'labels', 'weights', 'detailed_labels'}]
    X = df[feature_cols].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, labels, weights, test_size=0.3, random_state=42, stratify=labels
    )

    model = BoostedDecisionTree()
    model.fit(X_train, y_train, weights=w_train)
    scores = model.predict(X_test)

    S_mask = y_test == 1
    B_mask = y_test == 0

    mu_hat, sigma_mu, nll_min = mod.compute_mu_unbinned(
        1.0,
        scores[S_mask],
        w_test[S_mask],
        scores[B_mask],
        w_test[B_mask],
        scores,
        w_test,
    )

    payload = {
        'status': 'ok',
        'rows_used': int(len(df)),
        'test_events': int(len(X_test)),
        'mu_hat': float(mu_hat),
        'sigma_mu': float(sigma_mu),
        'nll_min': float(nll_min),
    }
except Exception as exc:
    payload = {
        'status': 'error',
        'error_type': type(exc).__name__,
        'error': str(exc),
        'traceback': traceback.format_exc(),
    }

out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
print(out_path)
