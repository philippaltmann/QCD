import sys
from evo.run import run_evo

res = []
print(f"Running GA Baseline for {sys.argv[1]}")
for seed in range(8): res.append(run_evo(sys.argv[1], seed))
for k in res[0].keys(): print(f"{k}: {sum([r[k] for r in res])/len(res)}")
