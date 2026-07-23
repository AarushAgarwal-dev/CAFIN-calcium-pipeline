import os, json
from cafin_pipeline import run

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Total data")
jobs = [
    ("Trial1 baseline (before drug)", "lat_trial1_beforedrug", "C2-MAX_latA_trial1_beforedrug", "C1-MAX_latA_trial1_beforedrug"),
    ("Trial1 after LatA",             "lat_trial1_afterdrug",  "AVG_C2-latA_trial1_afterdrug",  "AVG_C1-latA_trial1_afterdrug"),
    ("Trial2 baseline (before drug)", "lat_trial2_beforedrug", "AVG_C2-latA_trial2_beforedrug", "AVG_C1-latA_trial2_beforedrug"),
    ("Trial2 after LatA",             "lat_trial2_afterdrug",  "AVG_C2-latA_trial2_afterdrug",  "AVG_C1-latA_trial2_afterdrug"),
]
results = []
for label, folder, mb, cb in jobs:
    try:
        results.append(run(label, os.path.join(BASE, folder), mb, cb))
    except Exception as e:
        print(f"!! {label} FAILED: {e}")
        results.append({"label": label, "error": str(e)})

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_summary.json")
json.dump(results, open(out, "w"), indent=2)
print("\nSUMMARY ->", out)
for r in results:
    print(r)
