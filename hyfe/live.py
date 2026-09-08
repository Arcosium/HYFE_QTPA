"""실전 엔진(P7 페이퍼 트레이딩) — 매시간 12분에 실행. CryptoBars history + 오늘 버퍼로 최근 봉을 만들고,
저장된 GBM 4개(발생 15m·120·10 + 방향 15m·240·60 / 15m·120·60 / 1h·60·15)로 점수를 낸 뒤
방향 앙상블 하위 문턱(OS 10% 분위) 이하를 공매도(K 상한, 15h 보유, 편도 0.1%) 하는 페이퍼 포트폴리오를 굴린다.
usage: python3 -m hyfe.live tick [--dry]   |   python3 -m hyfe.live status
상태: work/live/state.json · 신호: work/live/signals.jsonl · 거래: work/live/trades.jsonl
"""
import argparse, glob, json, os, sys, time
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
from hyfe import bars as B, features as F, metrics as M

HIST = B.HIST; BUF = "/home/arcosium/projects/CryptoBars/data/bars"
LIVE = "work/live"; STATE = f"{LIVE}/state.json"
MODELS = {  # (res, W, H, stride) → 저장된 부스터(홀드아웃 학습본, IS ≤ 2025-11). train_final 로 갱신
    "event": ("15m", 120, 10, "work/results/hold_gbm_event_15m_W120_H10_s4"),
    "d1": ("15m", 240, 60, "work/results/hold_gbm_dir_15m_W240_H60_s4"),
    "d2": ("15m", 120, 60, "work/results/hold_gbm_dir_15m_W120_H60_s4"),
    "d3": ("1h", 60, 15, "work/results/hold_gbm_dir_1h_W60_H15_s4"),
}
K, HOLD_H, COST, Q = 20, 60, 0.001, 0.10          # 동시 보유, 보유 봉(15m), 편도 비용, 문턱 분위
MS15 = 15 * 60_000; MS4H = 4 * 3_600_000
# 방향 롱숏 장부(상대수익 relbin, 4h 격자, 상·하위 10% 를 K/2 씩 달러 중립, 보유 봉 뒤 청산). v1 = 홀드아웃 통과본, v2 = 롤링 선택 개선판(xs 피처+small/reg)
DIR_BOOKS = {"v1": {"res": "4h", "models": ["v1_4h_W60_H42", "v1_4h_W120_H42"], "hold": 42, "K": 100},
             "v2": {"res": "4h", "models": ["v2s_4h_W60_H84", "v2r_4h_W60_H42", "v2s_4h_W120_H42"], "hold": 84, "K": 100},
             "v2d": {"res": "1d", "models": ["v2d_1d_W20_H20"], "hold": 20, "K": 100},   # 1d 장부는 00:00 UTC 마감에만
             "cnn": {"res": "4h", "models": [], "cnn": ["final_i1_heatf_4h_W60_H84_sd0", "final_i1_heatf_4h_W60_H84_sd1", "final_i1_heatf_4h_W60_H84_sd2"], "hold": 84, "K": 100}}   # 이미지 CNN 장부(heatf, 시드 z-평균)


def load_buffer(days=2):
    """오늘·어제 버퍼(10분마다 flush) 전부 → base 별 DataFrame"""
    fs = []
    for d in sorted(glob.glob(f"{BUF}/date=*"))[-days:]:
        fs += sorted(glob.glob(f"{d}/*.parquet"))
    if not fs:
        return {}
    df = pd.concat(pd.read_parquet(f, columns=["ts", "base", "open", "high", "low", "close", "volume", "quote_volume"]) for f in fs)
    return {b: g for b, g in df.groupby("base")}


def recent_1m(base, buf, months=2):
    """history 최근 months 개월 + 버퍼 → 균일 1분 격자 (bars.load_1m 과 같은 규약)"""
    now = pd.Timestamp.utcnow()
    parts = [f"{HIST}/base={base}/part-{(now - pd.DateOffset(months=i)).strftime('%Y-%m')}.parquet" for i in range(months, -1, -1)]
    fr = [pd.read_parquet(p, columns=["ts", "open", "high", "low", "close", "volume", "quote_volume"]) for p in parts if os.path.exists(p)]
    if base in buf:
        fr.append(buf[base][["ts", "open", "high", "low", "close", "volume", "quote_volume"]])
    if not fr:
        return None
    df = pd.concat(fr).drop_duplicates("ts").sort_values("ts")
    df["quote_volume"] = df.quote_volume.fillna(df.close * df.volume)
    grid = np.arange(df.ts.iloc[0], df.ts.iloc[-1] + B.MS, B.MS)
    df = df.set_index("ts").reindex(grid); df["close"] = df.close.ffill()
    for c in ("open", "high", "low"):
        df[c] = df[c].fillna(df.close)
    df[["volume", "quote_volume"]] = df[["volume", "quote_volume"]].fillna(0.0)
    df.index.name = "ts"; return df.reset_index()


def latest_features(g1m, res, W, H, decision_ts):
    """판단 시각(봉 마감) 이 decision_ts 인 창의 피처 한 행. 없으면 None."""
    bars = B.to_res(g1m, B.RES_MIN[res]); bar_ms = B.RES_MIN[res] * 60_000
    bars = bars[bars.ts + bar_ms <= decision_ts]              # 마감된 봉만
    if bars is None or len(bars) < W + 2:
        return None
    b30 = max(W, 30 * 1440 // B.RES_MIN[res])                  # 30일 거래대금 맥락(qv_rel30) 에 필요한 길이
    bars = bars.tail(b30 + W + 5).reset_index(drop=True)
    # 라벨용 H 봉이 없으므로 미래 없이 피처만: make 는 e+H 를 요구 → 뒤에 가짜 봉 H 개를 붙여 계산하고 마지막 실봉 행만 쓴다
    pad = pd.concat([bars.iloc[[-1]]] * H, ignore_index=True); pad["ts"] = bars.ts.iloc[-1] + bar_ms * np.arange(1, H + 1)
    f = F.make(pd.concat([bars, pad], ignore_index=True), W, H, stride=1, bars_per_30d=b30)
    f = f[f.ts == bars.ts.iloc[-1]]
    return f.iloc[-1] if len(f) else None


ALARM_Q = 0.05     # 경보 = 발생 모델 급락·급등 확률 상위 5%(OS 분위)


def load_models():
    ms, thr = {}, {}
    for k, (res, W, H, stem) in MODELS.items():
        ms[k] = lgb.Booster(model_file=stem + "_model.txt")
        z = np.load(stem + "_ospred.npz", allow_pickle=True)
        if k == "event":
            thr["dn"] = float(np.quantile(z["p"][:, 2], 1 - ALARM_Q)); thr["up"] = float(np.quantile(z["p"][:, 1], 1 - ALARM_Q))
        else:
            thr[k] = float(np.quantile(M.score(z["p"]), Q))
    return ms, thr


def liquid_bases(n=200):
    u = pd.read_csv("work/universe.csv"); u = u[u.exclude.fillna("") == ""]
    from hyfe.pilot_gbm import liq_top
    os.environ.setdefault("HYFE_BARS", "work/bars_hold")
    return sorted(liq_top(u.base.tolist(), pd.Timestamp.utcnow().strftime("%Y-%m"), 12, n))


def state_load():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"equity": 1.0, "positions": {}, "n_trades": 0, "started": pd.Timestamp.utcnow().isoformat()}


def tick(dry=False):
    os.makedirs(LIVE, exist_ok=True)
    now = pd.Timestamp.utcnow(); decision = int(now.floor("h").value // 1_000_000)   # 이번 정시 = 판단 시각
    ms, thr = load_models(); buf = load_buffer(); st = state_load()
    bases = liquid_bases()
    rows = []
    for b in bases:
        g = recent_1m(b, buf)
        if g is None or g.ts.iloc[-1] < decision - 20 * 60_000:      # 20분 넘게 끊긴 종목은 건너뜀
            continue
        r = {"base": b, "ts": decision}
        ok = True
        for k, (res, W, H, _) in MODELS.items():
            f = latest_features(g, res, W, H, decision)
            if f is None:
                ok = False; break
            p = ms[k].predict(f[F.FEATURES].to_numpy(dtype=float)[None, :])[0]
            r[f"{k}_up"], r[f"{k}_dn"] = float(p[1]), float(p[2])
        if not ok:
            continue
        r["dir"] = float(np.mean([r[f"d{i}_up"] - r[f"d{i}_dn"] for i in (1, 2, 3)]))
        r["ev"] = r["event_up"] + r["event_dn"]
        # 진입가 = 판단 시각 직후 15m 봉 시가(= 판단 시각의 1분봉 시가), 청산은 60봉 뒤
        i = np.searchsorted(g.ts.to_numpy(), decision)
        r["px_open"] = float(g.open.iloc[i]) if i < len(g) and g.ts.iloc[i] == decision else None
        r["close_now"] = float(g.close.iloc[-1])
        rows.append(r)
    sig = pd.DataFrame(rows)
    if sig.empty:
        print("no signals (data?)"); return
    thr_dir = float(np.mean([thr[k] for k in ("d1", "d2", "d3")]))
    # ── 경보(주력): 급락·급등 확률 상위 5% ──
    al = sig[(sig.event_dn >= thr["dn"]) | (sig.event_up >= thr["up"])]
    with open(f"{LIVE}/alarms.jsonl", "a") as fh:
        for _, r in al.iterrows():
            fh.write(json.dumps({"ts": decision, "base": r.base, "p_dn": round(r.event_dn, 4), "p_up": round(r.event_up, 4), "kind": "dn" if r.event_dn >= thr["dn"] else "up",
                                 "px": r.close_now, "sigH": None}) + "\n")
    # 2.5h(10봉) 지난 경보의 실제 결과 대조 → alarm_outcomes.jsonl (급락/급등 = 2σ_H 기준은 사후 봉으로 계산)
    resolve_alarms(decision, buf)
    # ── 실험 장부 2: 대칭 롱숏(급등 상위5% 롱·급락 상위5% 숏, 2.5h, 각 K/2) — 백테스트에선 비용 미달, 실측으로 재검증 ──
    ls = st.setdefault("ls", {"equity_taker": 1.0, "equity_maker": 1.0, "positions": {}, "n_trades": 0})
    for b in list(ls["positions"]):
        pos = ls["positions"][b]
        if pos["exit_ts"] <= decision:
            g = recent_1m(b, buf); j = np.searchsorted(g.ts.to_numpy(), pos["exit_ts"])
            if j < len(g):
                px = float(g.close.iloc[max(j - 1, 0)]); ret = (px / pos["entry"] - 1) * (1 if pos["side"] == "long" else -1)
                ls["equity_taker"] += pos["size"] * (ret - 2 * COST); ls["equity_maker"] += pos["size"] * (ret - 2 * 0.0002); ls["n_trades"] += 1; ls["positions"].pop(b)
                with open(f"{LIVE}/ls_trades.jsonl", "a") as fh:
                    fh.write(json.dumps({**pos, "base": b, "exit_px": px, "ret": ret, "eq_taker": ls["equity_taker"], "eq_maker": ls["equity_maker"]}) + "\n")
    half = K // 2
    for sd, col, thr_k in (("long", "event_up", "up"), ("short", "event_dn", "dn")):
        n_sd = sum(1 for v in ls["positions"].values() if v["side"] == sd)
        cand2 = sig[(sig[col] >= thr[thr_k]) & sig.px_open.notna()].sort_values(col, ascending=False)
        for _, r in cand2.iterrows():
            if n_sd >= half:
                break
            if r.base in ls["positions"]:
                continue
            ls["positions"][r.base] = {"side": sd, "entry_ts": decision, "exit_ts": decision + 10 * MS15, "entry": r.px_open, "size": 1.0 / K, "p": float(r[col])}; n_sd += 1
    # 1) 만기 청산
    for b in list(st["positions"]):
        pos = st["positions"][b]
        if pos["exit_ts"] <= decision:
            g = recent_1m(b, buf); j = np.searchsorted(g.ts.to_numpy(), pos["exit_ts"])
            if j < len(g):
                px = float(g.close.iloc[max(j - 1, 0)]); ret = 1 - px / pos["entry"]; pnl = pos["size"] * (ret - 2 * COST)
                st["equity"] += pnl; st["n_trades"] += 1; st["positions"].pop(b)
                with open(f"{LIVE}/trades.jsonl", "a") as fh:
                    fh.write(json.dumps({**pos, "base": b, "exit_px": px, "ret": ret, "pnl": pnl, "equity": st["equity"]}) + "\n")
    # 2) 신규 공매도: 방향 점수 낮은 순, 문턱 이하, K 상한
    cand = sig[(sig.dir <= thr_dir) & sig.px_open.notna()].sort_values("dir")
    opened = []
    for _, r in cand.iterrows():
        if len(st["positions"]) >= K:
            break
        if r.base in st["positions"]:
            continue
        st["positions"][r.base] = {"entry_ts": decision, "exit_ts": decision + HOLD_H * MS15, "entry": r.px_open, "size": st["equity"] / K, "dir": r.dir}
        opened.append(r.base)
    with open(f"{LIVE}/signals.jsonl", "a") as fh:
        fh.write(json.dumps({"ts": decision, "n": len(sig), "thr": thr_dir, "opened": opened, "n_open": len(st["positions"]), "equity": st["equity"],
                             "top_event": sig.nlargest(5, "ev")[["base", "ev"]].values.tolist(), "bottom_dir": sig.nsmallest(5, "dir")[["base", "dir"]].values.tolist()}) + "\n")
    for res_, period in (("4h", MS4H), ("1d", 24 * 3_600_000)):
        if decision % period == 0:
            try:
                dir_tick(decision, buf, bases, st, sig, res_)
            except Exception as e:
                print(f"[dir {res_}] error", repr(e)[:200])
    print(f"{now:%Y-%m-%d %H:%M} UTC  종목 {len(sig)}  경보 {len(al)}건(급락 {(al.event_dn >= thr['dn']).sum()} 급등 {(al.event_up >= thr['up']).sum()})  [실험 공매도] 문턱 {thr_dir:+.4f} 후보 {len(cand)} 신규 {len(opened)} 보유 {len(st['positions'])} 순자산 {st['equity']:.4f}")
    if not dry:
        json.dump(st, open(STATE, "w"), indent=1)


def dir_tick(decision, buf, bases, st, sig, res="4h"):
    """봉 마감 시각마다(res 4h: UTC 0·4·8·…시, 1d: 0시): 창 피처 → 횡단면 순위 → 저장 모델 점수 z-평균 → 장부별 롱숏 페이퍼 거래."""
    from hyfe.pilot_gbm import add_xs
    books = {k: v for k, v in DIR_BOOKS.items() if v["res"] == res}
    metas = {}
    for bk in books.values():
        for m in bk["models"]:
            if m not in metas and os.path.exists(f"work/models/{m}_meta.json"):
                metas[m] = (json.load(open(f"work/models/{m}_meta.json")), lgb.Booster(model_file=f"work/models/{m}_model.txt"))
    if not metas:
        return
    Ws = sorted({int(m.split("_W")[1].split("_")[0]) for m in metas}); bar_ms = B.RES_MIN[res] * 60_000
    rows = []
    for b in bases:
        g = recent_1m(b, buf, months=2 if res == "4h" else 6)
        if g is None or g.ts.iloc[-1] < decision - 20 * 60_000:
            continue
        r = {"base": b, "ts": decision}
        for W in Ws:
            f = latest_features(g, res, W, 84 if res == "4h" else 20, decision)
            if f is None:
                r = None; break
            r[W] = f
        if r is None:
            continue
        i = np.searchsorted(g.ts.to_numpy(), decision)
        r["px_open"] = float(g.open.iloc[i]) if i < len(g) and g.ts.iloc[i] == decision else None
        rows.append(r)
    if len(rows) < 40:
        print(f"[dir {res}] 종목 {len(rows)} 부족 — 건너뜀"); return
    feat = {W: add_xs(pd.DataFrame([r[W] for r in rows]).assign(ts=decision)) for W in Ws}
    sc = pd.DataFrame({"base": [r["base"] for r in rows], "px_open": [r["px_open"] for r in rows]})
    for m, (meta, booster) in metas.items():
        W = int(m.split("_W")[1].split("_")[0]); X = feat[W][meta["features"]].to_numpy(dtype=float)
        p = booster.predict(X); sc[m] = (p[:, 1] - p[:, 2] - meta["score_mean"]) / max(meta["score_std"], 1e-9)
    for name, bk in books.items():   # 이미지 CNN 장부: 같은 4h 봉·피처(W=60)로 heatf 를 그려 CNN 시드들의 점수를 시각별 z-평균
        if not bk.get("cnn"):
            continue
        try:
            from hyfe import live_cnn as LC
            stems = [f"work/models/{m}" for m in bk["cnn"] if os.path.exists(f"work/models/{m}.pt") and os.path.exists(f"work/models/{m}.json")]
            if not stems or 60 not in feat:
                continue
            models = [LC.load_model(st_) for st_ in stems]
            bb = {}
            for r in rows:
                g = recent_1m(r["base"], buf, months=2); bars_ = B.to_res(g, 240); bb[r["base"]] = bars_[bars_.ts + MS4H <= decision]
            ctx = LC.bar_context(bb); fe = feat[60].set_index(pd.Index([r["base"] for r in rows]))
            cols = np.zeros(len(rows))
            for m_, meta_ in models:
                imgs = [LC.render(ctx, r["base"], decision, fe.loc[r["base"]], meta_) for r in rows]
                ok = np.array([im is not None for im in imgs]); s_ = np.full(len(rows), np.nan)
                if ok.any():
                    s_[ok] = LC.score(m_, torch.cat([im for im in imgs if im is not None]))
                z_ = (s_ - np.nanmean(s_)) / (np.nanstd(s_) + 1e-9); cols = cols + np.nan_to_num(z_)
            sc[name + "_score"] = cols / len(models); bk["models"] = [name + "_score"]
        except Exception as e:
            print("[dir cnn] error", repr(e)[:200])
    dirs = st.setdefault("dir", {})
    for name, bk in books.items():
        if any(m not in sc for m in bk["models"]):
            continue
        sc[name] = sc[bk["models"]].mean(axis=1)
        book = dirs.setdefault(name, {"equity": 1.0, "positions": {}, "n_trades": 0, "started": pd.Timestamp.utcnow().isoformat()})
        for b in list(book["positions"]):                     # 만기 청산(종가)
            pos = book["positions"][b]
            if pos["exit_ts"] <= decision:
                g = recent_1m(b, buf); j = np.searchsorted(g.ts.to_numpy(), pos["exit_ts"])
                if j < len(g):
                    px = float(g.close.iloc[max(j - 1, 0)]); ret = (px / pos["entry"] - 1) * (1 if pos["side"] == "long" else -1)
                    pnl = pos["size"] * (ret - 2 * COST); book["equity"] += pnl; book["n_trades"] += 1; book["positions"].pop(b)
                    with open(f"{LIVE}/dir_{name}_trades.jsonl", "a") as fh:
                        fh.write(json.dumps({**pos, "base": b, "exit_px": px, "ret": ret, "pnl": pnl, "equity": book["equity"]}) + "\n")
        half = bk["K"] // 2; q = sc[name].quantile([0.1, 0.9]); opened = []
        for side, cand in (("long", sc[sc[name] >= q[0.9]].sort_values(name, ascending=False)), ("short", sc[sc[name] <= q[0.1]].sort_values(name))):
            n_sd = sum(1 for v in book["positions"].values() if v["side"] == side)
            for _, r in cand.iterrows():
                if n_sd >= half:
                    break
                if r.base in book["positions"] or r.px_open is None or not np.isfinite(r.px_open):
                    continue
                book["positions"][r.base] = {"side": side, "entry_ts": decision, "exit_ts": decision + bk["hold"] * bar_ms, "entry": float(r.px_open), "size": book["equity"] / bk["K"], "score": float(r[name])}
                n_sd += 1; opened.append(f"{side[0]}:{r.base}")
        with open(f"{LIVE}/dir_{name}_signals.jsonl", "a") as fh:
            fh.write(json.dumps({"ts": decision, "n": len(sc), "opened": opened, "n_open": len(book["positions"]), "equity": book["equity"],
                                 "top": sc.nlargest(5, name)[["base", name]].round(3).values.tolist(), "bottom": sc.nsmallest(5, name)[["base", name]].round(3).values.tolist()}) + "\n")
        print(f"[dir {name}] 종목 {len(sc)} 신규 {len(opened)} 보유 {len(book['positions'])} 순자산 {book['equity']:.4f}")


def resolve_alarms(decision, buf, H=10):
    """경보 시각 + H 봉(2.5h) 이 지난 경보에 실제 15m 수익률·라벨을 붙인다(한 번만)."""
    p = f"{LIVE}/alarms.jsonl"; q = f"{LIVE}/alarm_outcomes.jsonl"
    if not os.path.exists(p):
        return
    done = set()
    if os.path.exists(q):
        done = {(json.loads(l)["ts"], json.loads(l)["base"]) for l in open(q) if l.strip()}
    out = []
    for l in open(p):
        a = json.loads(l)
        if (a["ts"], a["base"]) in done or a["ts"] + (H + 1) * MS15 > decision:
            continue
        g = recent_1m(a["base"], buf)
        if g is None:
            continue
        b = B.to_res(g, 15); i = np.searchsorted(b.ts.to_numpy(), a["ts"]) - 1        # 경보 시각 직전(마감된) 봉 = 창 끝
        if i < 130 or i + H >= len(b):
            continue
        c = b.c.to_numpy(); r1 = np.diff(np.log(np.maximum(c, 1e-12)), prepend=0)
        sig1 = float(np.std(r1[i - 119:i + 1])); sigH = sig1 * np.sqrt(H); fwd = float(np.log(c[i + H] / c[i]))
        label = 1 if fwd > 2 * sigH else (2 if fwd < -2 * sigH else 0)
        out.append({**a, "fwd": fwd, "sigH": sigH, "label": label, "hit": int((a["kind"] == "dn" and label == 2) or (a["kind"] == "up" and label == 1)), "absmove_z": abs(fwd) / max(sigH, 1e-9)})
    if out:
        with open(q, "a") as fh:
            for o in out:
                fh.write(json.dumps(o) + "\n")


def status():
    st = state_load(); print(json.dumps({k: v for k, v in st.items() if k != "positions"}, indent=1)); print("positions", len(st["positions"]))
    if os.path.exists(f"{LIVE}/trades.jsonl"):
        t = pd.read_json(f"{LIVE}/trades.jsonl", lines=True); print(f"[실험용 공매도 장부] trades {len(t)} mean ret {t.ret.mean()*1e4:+.1f}bp hit {(t.pnl>0).mean():.2f}")
    ls = st.get("ls")
    if ls:
        print(f"[실험 대칭 롱숏] trades {ls['n_trades']} 보유 {len(ls['positions'])} 순자산 taker {ls['equity_taker']:.4f} / maker {ls['equity_maker']:.4f}")
    for name, bk in (st.get("dir") or {}).items():
        line = f"[방향 롱숏 {name}] trades {bk['n_trades']} 보유 {len(bk['positions'])} 순자산 {bk['equity']:.4f} 시작 {bk.get('started','')[:16]}"
        if os.path.exists(f"{LIVE}/dir_{name}_trades.jsonl"):
            t = pd.read_json(f"{LIVE}/dir_{name}_trades.jsonl", lines=True); line += f" | 거래당 {t.ret.mean()*1e4:+.1f}bp 적중 {(t.ret>0).mean():.2f} 롱 {t[t.side=='long'].ret.mean()*1e4:+.1f}bp 숏 {t[t.side=='short'].ret.mean()*1e4:+.1f}bp"
        print(line)
    if os.path.exists(f"{LIVE}/alarm_outcomes.jsonl"):
        o = pd.read_json(f"{LIVE}/alarm_outcomes.jsonl", lines=True)
        for k, g in o.groupby("kind"):
            print(f"[경보 {k}] n={len(g)} 적중률 {g.hit.mean():.3f} (기저 약 0.03) 절대변동/σ {g.absmove_z.mean():.2f} (전체 약 0.75)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["tick", "status"]); ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    tick(a.dry) if a.cmd == "tick" else status()
