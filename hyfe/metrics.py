"""평가 지표: 급등·급락 PR-AUC(유병률 대비 lift) + 경제 지표(점수 상·하위 10% 선행수익 스프레드) + 블록 부트스트랩 CI."""
import numpy as np
from sklearn.metrics import average_precision_score


def score(p):
    """3클래스 확률 → 방향 점수 (급등 − 급락)"""
    return p[:, 1] - p[:, 2]


def evaluate(y, p, fwd=None, sigH=None):
    r = {}
    for cls, name in ((1, "up"), (2, "dn")):
        yy = (y == cls).astype(int)
        r[f"ap_{name}"] = float(average_precision_score(yy, p[:, cls])) if 0 < yy.sum() < len(yy) else float("nan")
        r[f"prev_{name}"] = float(yy.mean())
    r["ap"] = float(np.nanmean([r["ap_up"], r["ap_dn"]])); r["prev"] = (r["prev_up"] + r["prev_dn"]) / 2
    r["lift"] = r["ap"] / max(r["prev"], 1e-9)
    if fwd is not None and len(fwd) >= 100:
        s = score(p); q = np.quantile(s, [0.1, 0.9])
        hi, lo = s >= q[1], s <= q[0]
        r["spread"] = float(fwd[hi].mean() - fwd[lo].mean())        # 상위10% − 하위10% 평균 선행 로그수익 (라벨 정의와 무관한 공통 잣대)
        if sigH is not None:
            z = fwd / np.maximum(sigH, 1e-9); r["spread_z"] = float(z[hi].mean() - z[lo].mean())
        r["hit_top"] = float((fwd[hi] > 0).mean()); r["hit_bot"] = float((fwd[lo] < 0).mean())   # 십분위 방향 적중률(50% 가 기준)
    return r


def paired_bootstrap(y, pa, pb, groups, n=1000, seed=0):
    """AP(a) − AP(b) 의 블록 부트스트랩 95% CI. groups = 블록 id(종목×월)."""
    rng = np.random.default_rng(seed)
    _, gidx = np.unique(groups, return_inverse=True)
    members = [np.where(gidx == i)[0] for i in range(gidx.max() + 1)]
    d = []
    for _ in range(n):
        ii = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        d.append(evaluate(y[ii], pa[ii])["ap"] - evaluate(y[ii], pb[ii])["ap"])
    d = np.array(d)
    return {"diff": evaluate(y, pa)["ap"] - evaluate(y, pb)["ap"], "lo": float(np.quantile(d, 0.025)),
            "hi": float(np.quantile(d, 0.975)), "p_le0": float((d <= 0).mean())}
