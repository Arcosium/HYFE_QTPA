"""Runpod pod 생성/조회/종료 (REST, Bearer). GraphQL 은 조회만. Cloudflare 가 기본 UA 를 막으므로 브라우저 UA 필수.
usage: python3 -m hyfe.runpod create [--gpu "NVIDIA A40"] [--name hyfe-s0] | status [id] | ssh [id] | terminate [id] | balance
상태 파일: work/pod.json (키는 vault 에만 있다)
"""
import argparse, json, os, sys, time, urllib.error, urllib.request

POD_FILE = "work/pod.json"
IMAGE = "runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2404"
REST = "https://rest.runpod.io/v1"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/128.0 Safari/537.36"


def _key():
    for l in open("/home/arcosium/vault/AI말평/.env", encoding="utf-8"):
        if l.startswith("RUNPOD_API_KEY="):
            return l.split("=", 1)[1].strip()
    raise RuntimeError("RUNPOD_API_KEY not in vault .env")


def _req(url, method="GET", body=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None, method=method,
                                 headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json", "User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = r.read().decode()
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {url} -> {e.code}: {e.read().decode()[:300]}")


def rest(method, path, body=None):
    return _req(REST + path, method, body)


def gql(query):
    return _req("https://api.runpod.io/graphql", "POST", {"query": query})


def balance():
    return gql("{ myself { clientBalance spendLimit } }")["data"]["myself"]


def create(gpu, name, disk, cloud, volume=0):
    pod = rest("POST", "/pods", {
        "name": name, "imageName": IMAGE, "cloudType": cloud, "gpuCount": 1, "gpuTypeIds": [gpu],
        "containerDiskInGb": disk, "volumeInGb": volume, "volumeMountPath": "/workspace",
        "ports": ["22/tcp"], "supportPublicIp": True})
    os.makedirs("work", exist_ok=True)
    json.dump(pod, open(POD_FILE, "w"), indent=2)
    return pod


def status(pod_id):
    return rest("GET", f"/pods/{pod_id}")


def ssh_endpoint(pod_id, wait_sec=900):
    t0 = time.time()
    while time.time() - t0 < wait_sec:
        p = status(pod_id)
        pm = p.get("portMappings") or {}
        if p.get("publicIp") and pm.get("22"):
            return p["publicIp"], pm["22"]
        time.sleep(15)
    raise TimeoutError("SSH 포트가 열리지 않음")


def saved():
    return json.load(open(POD_FILE))["id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["create", "status", "ssh", "terminate", "balance", "list"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--gpu", default="NVIDIA A40"); ap.add_argument("--name", default="hyfe")
    ap.add_argument("--disk", type=int, default=100); ap.add_argument("--cloud", default="COMMUNITY")
    a = ap.parse_args()
    if a.cmd == "balance":
        print(balance())
    elif a.cmd == "list":
        print(json.dumps([{k: p.get(k) for k in ("id", "name", "costPerHr", "desiredStatus", "publicIp", "portMappings")} for p in rest("GET", "/pods")], indent=1))
    elif a.cmd == "create":
        pod = create(a.gpu, a.name, a.disk, a.cloud)
        print({k: pod.get(k) for k in ("id", "name", "costPerHr", "machineId")})
        ip, port = ssh_endpoint(pod["id"])
        print(f"ssh -o StrictHostKeyChecking=accept-new -p {port} root@{ip}")
    elif a.cmd == "status":
        print(json.dumps(status(a.arg or saved()), indent=1, ensure_ascii=False)[:2000])
    elif a.cmd == "ssh":
        ip, port = ssh_endpoint(a.arg or saved())
        print(f"ssh -o StrictHostKeyChecking=accept-new -p {port} root@{ip}")
    elif a.cmd == "terminate":  # 다른 서비스의 pod 보호: 이름이 hyfe- 인 것만 종료
        pid = a.arg or saved()
        name = status(pid).get("name", "")
        if not name.startswith("hyfe"):
            sys.exit(f"REFUSED: {pid} ({name}) 는 이 프로젝트 pod 이 아님")
        rest("DELETE", f"/pods/{pid}"); print("terminated", pid, name)
        if os.path.exists(POD_FILE) and (not a.arg or a.arg == saved()):
            os.remove(POD_FILE)


if __name__ == "__main__":
    main()
