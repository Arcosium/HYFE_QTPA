"""A40 pod 함대: 여러 pod 를 만들고(SSH 대기·의존성·코드·종목표 설치) 목록을 work/fleet.json 에 둔다. 데이터는 seed pod 에서 pod 간 복사.
usage: python3 -m hyfe.fleet create N [--gpu "NVIDIA A40"] | list | setup | copy_from <seed_ip> <seed_port> | run <name> <script> | term [name|all]
"""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from hyfe import runpod as R

FLEET = "work/fleet.json"
SSH = "ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o ServerAliveInterval=30"


def load():
    return json.load(open(FLEET)) if os.path.exists(FLEET) else {}


def save(f):
    json.dump(f, open(FLEET, "w"), indent=1)


def sh(pod, cmd, timeout=1800):
    return subprocess.run(f"{SSH} -p {pod['port']} root@{pod['ip']} {json.dumps(cmd)}", shell=True, capture_output=True, text=True, timeout=timeout)


def create(n, gpu, cloud, prefix):
    f = load(); made = []
    nxt = max([int(k.split("-")[-1]) for k in f] + [0]) + 1
    for i in range(n):
        name = f"{prefix}{nxt + i}"
        try:
            pod = R.create(gpu, name, 60, cloud)
        except RuntimeError as e:
            print("create failed", name, str(e)[:200]); continue
        f[name] = {"id": pod["id"], "cost": pod.get("costPerHr")}; made.append(name); save(f)
        print("created", name, pod["id"], pod.get("costPerHr"))
    for name in made:
        ip, port = R.ssh_endpoint(f[name]["id"]); f[name].update(ip=ip, port=port); save(f)
        print("ssh", name, ip, port)
    return made


def setup(names):
    f = load()
    def one(name):
        p = f[name]
        r = sh(p, "python -m pip install --break-system-packages -q lightgbm scikit-learn pandas pyarrow pillow >/dev/null 2>&1; mkdir -p /workspace/hyfe/work/bars /workspace/hyfe/work/results; python -c 'import lightgbm,sklearn;print(1)'")
        subprocess.run(f"rsync -az -e '{SSH} -p {p['port']}' hyfe work/universe.csv work/liq12_s0.txt work/liq12_s1.txt work/liq12_s2.txt work/liq12_s3.txt work/liq12_s4.txt work/liqwf_all.txt root@{p['ip']}:/workspace/hyfe/ && rsync -az -e '{SSH} -p {p['port']}' work/results/teacher_v1.npz work/results/teacher_v2.npz root@{p['ip']}:/workspace/hyfe/work/results/ && {SSH} -p {p['port']} root@{p['ip']} 'mv -f /workspace/hyfe/universe.csv /workspace/hyfe/liq*.txt /workspace/hyfe/work/'", shell=True)
        return name, r.stdout.strip()[-1:] == "1"
    with ThreadPoolExecutor(8) as ex:
        for name, ok in ex.map(one, names):
            print("setup", name, "ok" if ok else "FAILED")


def copy_from(seed_ip, seed_port, names, key="/home/arcosium/vault/HYFE_QTPA/fleet_key"):
    """seed pod 의 work/bars 를 각 pod 로. 임시 키(vault)를 seed 의 authorized_keys 에 넣고 각 pod 가 seed 에서 rsync 로 당긴다."""
    f = load()
    if not os.path.exists(key):
        os.makedirs(os.path.dirname(key), exist_ok=True)
        subprocess.run(f"ssh-keygen -q -t ed25519 -N '' -f {key}", shell=True)
    pub = open(key + ".pub").read().strip()
    subprocess.run(f"{SSH} -p {seed_port} root@{seed_ip} \"grep -q '{pub.split()[1]}' ~/.ssh/authorized_keys || echo '{pub}' >> ~/.ssh/authorized_keys\"", shell=True)
    def one(name):
        p = f[name]
        subprocess.run(f"scp -o StrictHostKeyChecking=accept-new -P {p['port']} {key} root@{p['ip']}:/root/.ssh/fleet_key", shell=True)
        r = sh(p, f"chmod 600 /root/.ssh/fleet_key; rsync -a -e 'ssh -o StrictHostKeyChecking=accept-new -i /root/.ssh/fleet_key -p {seed_port}' root@{seed_ip}:/workspace/hyfe/work/bars/ /workspace/hyfe/work/bars/ && rsync -a -e 'ssh -o StrictHostKeyChecking=accept-new -i /root/.ssh/fleet_key -p {seed_port}' root@{seed_ip}:/workspace/hyfe/work/bars_hold/ /workspace/hyfe/work/bars_hold/ 2>/dev/null; du -sh /workspace/hyfe/work/bars | cut -f1", timeout=3600)
        return name, r.stdout.strip()
    with ThreadPoolExecutor(8) as ex:
        for name, out in ex.map(one, names):
            print("copy", name, out)


def run(name, script):
    """로컬 스크립트 파일을 pod 에 올려 nohup 으로 실행. 로그 work/results/<script>.log"""
    p = load()[name]; base = os.path.basename(script)
    subprocess.run(f"scp -q -P {p['port']} {script} root@{p['ip']}:/workspace/hyfe/{base}", shell=True)
    try:  # nohup 자식이 ssh 세션을 붙들어 ssh 가 안 끝난다 — 프로세스는 이미 떠 있으므로 20초 뒤 끊는다
        sh(p, f"cd /workspace/hyfe && setsid nohup bash {base} > work/results/{base}.log 2>&1 < /dev/null &", timeout=20)
    except subprocess.TimeoutExpired:
        pass
    print("launched", name, base)


def collect(names=None):
    """각 pod 의 work/results/ 를 로컬 work/results/ 로 합친다(이름이 유일하므로 그대로 병합)."""
    f = load()
    for name in names or list(f):
        p = f[name]
        subprocess.run(f"rsync -az -e '{SSH} -p {p['port']}' --exclude '*.pt' root@{p['ip']}:/workspace/hyfe/work/results/ work/results/", shell=True)


def tail(names=None, state="work/fleet_tail.json"):
    """pod 로그의 새 줄(===, DONE, 오류) 만 출력. 커서는 state 파일에."""
    f = load(); cur = json.load(open(state)) if os.path.exists(state) else {}
    for name in names or list(f):
        p = f[name]
        r = sh(p, "cat /workspace/hyfe/work/results/*.sh.log 2>/dev/null", timeout=60)
        lines = [l for l in r.stdout.splitlines() if any(k in l for k in ("===", "DONE", "Traceback", "Error", "insufficient", "Killed"))]
        for l in lines[cur.get(name, 0):]:
            print(f"[{name}] {l[:160]}", flush=True)
        cur[name] = len(lines)
    json.dump(cur, open(state, "w"))


def refresh():
    """API 에서 ip/port 를 채운다(아직 없는 pod 는 그대로). 준비된 pod 이름 목록을 돌려준다."""
    f = load(); ready = []
    by_id = {p["id"]: p for p in R.rest("GET", "/pods")}
    for name, v in f.items():
        p = by_id.get(v["id"], {}); pm = p.get("portMappings") or {}
        if p.get("publicIp") and pm.get("22"):
            v.update(ip=p["publicIp"], port=pm["22"]); ready.append(name)
        print(name, v["id"], p.get("desiredStatus"), v.get("ip"), v.get("port"))
    save(f); return ready


QUEUE = "work/queue.txt"       # 한 줄 = pod 에서 실행할 셸 명령. 데몬이 위에서부터 빼 간다
BUSY = r"hyfe\.(train_cnn|pilot_gbm)|hyfe/s[0-9]\.sh|bash (p|q)[0-9]"


def busy(pod):
    try:
        r = sh(pod, f"pgrep -fc '{BUSY}'", timeout=60)
    except subprocess.TimeoutExpired:
        return True
    if r.returncode not in (0, 1):      # ssh 오류 → 건드리지 않는다
        return True
    return (r.stdout.strip() or "0") != "0"


def pop_queue():
    if not os.path.exists(QUEUE):
        return None
    lines = [l for l in open(QUEUE).read().splitlines() if l.strip() and not l.startswith("#")]
    if not lines:
        return None
    open(QUEUE, "w").write("\n".join(lines[1:]) + ("\n" if lines[1:] else ""))
    return lines[0]


def terminate(name):
    f = load(); pid = f[name]["id"]
    live = {p["id"]: p["name"] for p in R.rest("GET", "/pods")}
    if pid in live and not live[pid].startswith("hyfe-"):
        print("REFUSED (not ours):", name, live[pid], flush=True); return
    if pid in live:
        R.rest("DELETE", f"/pods/{pid}")
    f.pop(name); save(f); print("terminated", name, pid, flush=True)


def gpu_util(pod):
    try:
        r = sh(pod, "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits", timeout=30)
        return int(r.stdout.strip().splitlines()[0]) if r.returncode == 0 and r.stdout.strip() else -1
    except Exception:
        return -1


def queue_len():
    return len([l for l in open(QUEUE).read().splitlines() if l.strip() and not l.startswith("#")]) if os.path.exists(QUEUE) else 0


def scale_up(gpu="NVIDIA A40", cloud="SECURE"):
    """대기열이 남았는데 놀 pod 가 없으면 1대 증설: 생성 → 설치 → 데이터는 살아 있는 pod 에서 복사."""
    f = load(); seed = next((v for v in f.values() if v.get("ip")), None)
    made = create(1, gpu, cloud, "hyfe-")
    if not made:
        return
    setup(made)
    if seed:
        copy_from(seed["ip"], seed["port"], made)
    print(time.strftime("%H:%M"), "scaled up", made, flush=True)


def daemon(interval=60, max_pods=10):
    """1분마다: idle pod → 결과 수거 → 대기열 다음 명령 배정, 대기열 비면 즉시 종료(GPU 를 놀리지 않는다).
    대기열이 남았는데 idle pod 가 없으면 상한까지 증설. 세션과 무관하게 GB10 에서 돈다."""
    n = 0; cycle = 0
    while True:
        cycle += 1
        f = load()
        if not f:
            if queue_len():
                scale_up(); continue
            print(time.strftime("%H:%M"), "fleet empty, queue empty, exit", flush=True); return
        idle = 0; utils = []
        for name in list(f):
            p = f[name]
            if not p.get("ip"):
                continue
            try:
                if busy(p):
                    utils.append(f"{name}:{gpu_util(p)}%"); continue
                idle += 1
                collect([name])
                cmd = pop_queue()
                if cmd:
                    n += 1; path = f"work/pods/q{n}_{name}.sh"; open(path, "w").write(cmd + "\n")
                    run(name, path); print(time.strftime("%H:%M"), name, "<-", cmd[:100], flush=True)
                else:
                    terminate(name)
            except Exception as e:
                print(time.strftime("%H:%M"), name, "error", str(e)[:200], flush=True)
        if utils and cycle % 10 == 0:
            print(time.strftime("%H:%M"), "util", " ".join(utils), flush=True)
        if queue_len() and idle == 0 and len(load()) < max_pods:
            try:
                scale_up()
            except Exception as e:
                print(time.strftime("%H:%M"), "scale_up error", str(e)[:200], flush=True)
        time.sleep(interval)


def main():
    a = sys.argv[1:]
    cmd = a[0]
    if cmd == "daemon":
        daemon(max_pods=int(os.environ.get("HYFE_MAX_PODS", "10"))); return
    if cmd == "refresh":
        print("ready:", refresh()); return
    if cmd == "collect":
        collect(a[1:] or None); return
    if cmd == "tail":
        tail(a[1:] or None); return
    if cmd == "create":
        n = int(a[1]); gpu = a[a.index("--gpu") + 1] if "--gpu" in a else "NVIDIA A40"
        cloud = a[a.index("--cloud") + 1] if "--cloud" in a else "SECURE"
        made = create(n, gpu, cloud, "hyfe-"); setup(made)
    elif cmd == "list":
        for k, v in load().items():
            print(k, v)
    elif cmd == "setup":
        setup(a[1:] or list(load()))
    elif cmd == "copy_from":
        copy_from(a[1], a[2], a[3:] or list(load()))
    elif cmd == "run":
        run(a[1], a[2])
    elif cmd == "term":  # 다른 서비스의 pod 보호: fleet.json 에 있고 이름이 hyfe- 인 것만
        f = load(); live = {p["id"]: p["name"] for p in R.rest("GET", "/pods")}
        for k in (list(f) if a[1] == "all" else a[1:]):
            pid = f[k]["id"]
            if pid in live and not live[pid].startswith("hyfe-"):
                print("REFUSED (not ours):", k, live[pid]); continue
            R.rest("DELETE", f"/pods/{pid}"); print("terminated", k); f.pop(k)
        save(f)


if __name__ == "__main__":
    main()
