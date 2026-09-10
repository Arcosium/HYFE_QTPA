"""확장 실험 러너(9/10): 일감 파일 줄 수만큼 A40 pod 를 만들고, 봉 캐시(4h+1h, 폴드용+홀드아웃용)는 GB10→첫 pod 한 번만 올린 뒤
pod 끼리 복사(fleet.copy_from)해 준비한다. 준비된 pod 에 한 줄씩 배정하고 본 fleet.json 에 합친다. 남는 일감은 work/queue.txt 로(데몬이 배정).
usage: PYTHONPATH=. python3 work/ext_run.py work/ext_jobs.txt   (재실행하면 임시 fleet 의 pod 를 이어서 준비한다 — 새로 만들지 않음)
끝나면 HYFE_MAX_PODS=1 데몬이 결과 수거·종료."""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from hyfe import fleet

jobs = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
fleet.FLEET = "work/fleet_new.json"
f = fleet.load()
if not f:
    fleet.create(len(jobs), "NVIDIA A40", "SECURE", "hyfe-"); f = fleet.load()
names = [n for n in f if f[n].get("ip")]
for n in names:
    subprocess.run(f"ssh-keygen -R '[{f[n]['ip']}]:{f[n]['port']}'", shell=True, capture_output=True)   # Runpod 이 ip:port 를 재사용해 stale 호스트키가 접속을 막는다
fleet.setup(names)
seed = f[names[0]]; R = f"-e '{fleet.SSH} -p {seed['port']}' root@{seed['ip']}:/workspace/hyfe/work"
for src, dst in (("work/bars_hold/4h", "bars_hold/4h"), ("work/bars_full/4h", "bars/4h"), ("work/bars_hold/1h", "bars_hold/1h"), ("work/bars_full/1h", "bars/1h")):
    r = subprocess.run(f"rsync -az --rsync-path='mkdir -p /workspace/hyfe/work/{dst} && rsync' {src}/ {R}/{dst}/", shell=True, capture_output=True, text=True)
    print("seed ship", dst, "ok" if r.returncode == 0 else r.stderr[-200:], flush=True)
fleet.copy_from(seed["ip"], seed["port"], names[1:])
def verify(n):
    v = fleet.sh(f[n], "ls /workspace/hyfe/work/bars/4h | wc -l; ls /workspace/hyfe/work/bars/1h | wc -l; ls /workspace/hyfe/work/bars_hold/4h | wc -l; python -c 'import torch,lightgbm;print(1)'", timeout=120)
    w = v.stdout.split(); return n, len(w) == 4 and w[3] == "1" and min(int(x) for x in w[:3]) > 700
with ThreadPoolExecutor(8) as ex:
    ok = [n for n, good in ex.map(verify, names) if good or print("BAD", n, flush=True)]
os.makedirs("work/pods", exist_ok=True)
def go(j):
    n, cmd = j; path = f"work/pods/q{400 + int(n.split('-')[1])}_{n}.sh"; open(path, "w").write(cmd + "\n"); fleet.run(n, path); return j
with ThreadPoolExecutor(5) as ex:
    for n, cmd in ex.map(go, list(zip(ok, jobs))): print("dispatched", n, cmd[:70], flush=True)
if len(jobs) > len(ok):
    open("work/queue.txt", "a").write("\n".join(jobs[len(ok):]) + "\n"); print("queued", len(jobs) - len(ok), flush=True)
merged = json.load(open("work/fleet.json")) if os.path.exists("work/fleet.json") else {}
merged.update({n: f[n] for n in ok}); json.dump(merged, open("work/fleet.json", "w"), indent=1)
print("PODS_DISPATCHED", min(len(ok), len(jobs)), "of", len(jobs), "bad", [n for n in names if n not in ok], flush=True)
