"""창 학습기 — 표현·모델 4종을 같은 창·라벨·분할로 비교한다. 이미지는 GPU 에서 즉석 렌더.
  i1  JKX형 소형 CNN(이미지)          i2  ImageNet ResNet-18 파인튜닝(이미지)
  j2  정규화 시퀀스 W×5 → 소형 트랜스포머   f1  i1 특징 + 밀도 피처 late fusion
usage: python3 -m hyfe.train_cnn --res 15m --W 120 --H 10 --model i1 --top 200 --val 2025-09 --test 2025-12 --test_end 2026-03 --out work/results/x.json
  --no_volume 거래량 패널 제거 · --shuffle 라벨 셔플(플라시보) · --render bar|candle|line · --label ksigma|fixed|binary
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as Fn
from hyfe import bars as B, features as F, metrics as M
from hyfe.pilot_gbm import XS_FEATS

IMG_H = 96          # 전체 높이(JKX 60일 이미지와 같음). 거래량 패널은 이 안의 하단 18px
CHANNEL_SETS = {"gray": ["img"], "c2": ["img", "dens"], "c3": ["img", "dens", "hour"], "multi": ["img", "dens", "vrel", "hour", "dow"],
                "c3m": ["img", "dens", "hour", "btc_r24", "breadth24"], "multim": ["img", "dens", "vrel", "hour", "dow", "btc_r24", "btc_r7d", "breadth24"]}   # m = BTC 매크로 평면(사장 제안)
MACRO_COLS = ["btc_r24", "btc_r7d", "breadth24"]
VOL_H = 18
PX = 3              # 봉당 폭


def ms(month):
    return int(pd.Timestamp(month).value // 1_000_000)


# ---------- 데이터 ----------
class Windows:
    """종목별 봉 배열을 GPU 에 올려두고 (창 끝 g) 로 창을 뽑아 렌더/정규화한다. need_feats 면 밀도 피처도 붙인다."""

    def __init__(self, bases, res, W, H, k, device, start="2023-01", end="2026-02", label="ksigma", fixed=0.02, need_feats=False, stride=None, macro=False):
        self.W, self.H, self.dev = W, H, device
        B.build(bases, start, end, res_list=[res])
        arrs, rows, xs_rows = [], [], []
        rel = label.startswith("rel"); base_mode = "ksigma" if rel else label
        mt = pd.read_parquet(f"work/macro_{res}.parquet")[["ts"] + MACRO_COLS] if macro else None
        for b in bases:
            p = B.path(res, b)
            if not os.path.exists(p):
                continue
            df = pd.read_parquet(p)
            if mt is not None:
                df = df.merge(mt, on="ts", how="left").fillna({c: 0.0 for c in MACRO_COLS})
            lw = F.label_windows(df, W, H, k, stride=stride, mode=base_mode, fixed=fixed, align=bool(stride))
            if lw is None or not len(lw):
                continue
            if need_feats:
                ft = F.make(df, W, H, k, stride=stride, bars_per_30d=max(W, 30 * 1440 // B.RES_MIN[res]), mode=label, fixed=fixed, align=bool(stride))
                lw = lw.merge(ft[["ts"] + F.FEATURES], on="ts", how="inner")
            off = sum(len(a) for a in arrs)
            bpd = 1440 // B.RES_MIN[res]
            v30 = df.v.rolling(30 * bpd, min_periods=bpd).mean().bfill().to_numpy()
            dt = pd.to_datetime(df.ts, unit="ms")
            ctx = np.c_[np.log1p(df.v.to_numpy() / np.maximum(v30, 1e-9)), dt.dt.hour.to_numpy() / 23.0, dt.dt.dayofweek.to_numpy() / 6.0].astype(np.float32)
            mac = (np.c_[df.btc_r24.to_numpy() / 0.05, df.btc_r7d.to_numpy() / 0.15, df.breadth24.to_numpy() - 0.5] if mt is not None else np.zeros((len(df), 3))).astype(np.float32)
            arrs.append(np.c_[df[["o", "h", "l", "c", "v"]].to_numpy(np.float32), ctx, mac])   # 열: o h l c v vrel30 hour dow btc_r24 btc_r7d breadth24 (+ xs 3열은 아래서)
            lc = np.log(df.c.clip(lower=1e-12).to_numpy()); r1 = np.diff(lc, prepend=lc[0])
            xs_rows.append(pd.DataFrame({"ts": df.ts.to_numpy(), "r24": lc - np.r_[np.full(bpd, lc[0]), lc[:-bpd]], "vol24": pd.Series(r1).rolling(bpd, min_periods=2).std().bfill().to_numpy(), "vrel": ctx[:, 0]}))
            lw["base"] = b; lw["g"] = lw.e + off       # 전체 배열에서의 창 끝 위치
            rows.append(lw)
        arr = np.concatenate(arrs)
        xs = pd.concat(xs_rows, ignore_index=True)                     # 같은 봉 시각의 유니버스 안 백분위 순위 3열(heatx 렌더용) — 이미지가 못 보는 횡단면 정보
        g_ = xs.groupby("ts")
        xr = np.c_[g_.r24.rank(pct=True).to_numpy(), g_.vol24.rank(pct=True).to_numpy(), g_.vrel.rank(pct=True).to_numpy()].astype(np.float32)
        self.arr = torch.from_numpy(np.c_[arr, xr]).to(device)
        self.idx = pd.concat(rows, ignore_index=True)
        if rel:   # 횡단면 상대 수익 라벨: 같은 판단 시각 유니버스 평균을 뺀다(정렬 격자 필요)
            cnt = self.idx.groupby("ts").fwd.transform("size"); self.idx = self.idx[cnt >= 20].copy()
            self.idx["fwd"] = self.idx.fwd - self.idx.groupby("ts").fwd.transform("mean")
            self.idx["label"] = F.label_of(self.idx.fwd.to_numpy(), self.idx.sigH.to_numpy(), k, "binary" if label == "relbin" else "ksigma", fixed)

    def window(self, g):
        ar = torch.arange(self.W, device=self.dev)
        return self.arr[(g[:, None] - self.W + 1 + ar[None, :])]          # (B,W,11): o h l c v vrel30 hour dow btc_r24 btc_r7d breadth24

    def sequence(self, g, ctx=False):
        """j2 입력: 가격은 창 마지막 종가 대비 로그비×100, 거래량은 창 평균 대비. ctx=True 면 30일 대비 거래량·시각·요일 열 추가(8열)."""
        w = self.window(g); win = w[..., :5]; px = win[..., :4]; v = win[..., 4:5]
        px = torch.log(px.clamp_min(1e-9) / px[:, -1:, 3:4].clamp_min(1e-9)) * 100
        v = v / v.mean(1, keepdim=True).clamp_min(1e-9)
        cols = [px, v]
        if ctx:
            cols += [w[..., 5:6].clamp(-3, 3) / 3, w[..., 6:7] / 23.0, w[..., 7:8] / 6.0]
        return torch.cat(cols, -1)

    def heatx(self, g):
        """heat 8행 + 횡단면 순위 3행(24h 수익·24h 변동·30일 대비 거래량의 시각별 백분위) = 11행 열지도."""
        w = self.window(g); x = self.sequence(g, True)
        px = (x[..., :4] / 10.0).clamp(-1, 1) * 0.5 + 0.5; v = (x[..., 4:5] / 3.0).clamp(0, 1)
        img = torch.cat([px, v, x[..., 5:6] * 0.5 + 0.5, x[..., 6:8], w[..., 11:14]], -1).transpose(1, 2)   # (B,11,W)
        n = img.shape[1]; img = img.repeat_interleave(IMG_H // n, 1).repeat_interleave(PX, 2)
        pad = IMG_H - img.shape[1]
        if pad > 0:
            img = torch.cat([img, torch.zeros(img.shape[0], pad, img.shape[2], device=img.device)], 1)
        return img[:, None]

    def heatf(self, g, f):
        """GBM 입력을 그대로 이미지로: heatx 11행(시계열 열지도+순위) + 창 끝 피처 37행(표준화→시그모이드, 가로로 펼침) = 48행 → 96px."""
        hx = self.heatx(g)[:, 0]                                            # (B,IMG_H,Wp) — 11행이 IMG_H//11 배로 늘어나 있음
        rows = hx[:, ::(IMG_H // 11)][:, :11]                               # 원래 11행으로 되돌림
        fr = torch.sigmoid(f / 2.0)[:, :, None].expand(-1, -1, hx.shape[-1])  # (B,37,Wp)
        img = torch.cat([rows, fr], 1); n = img.shape[1]
        img = img.repeat_interleave(max(IMG_H // n, 1), 1); pad = IMG_H - img.shape[1]
        if pad > 0:
            img = torch.cat([img, torch.zeros(img.shape[0], pad, img.shape[2], device=img.device)], 1)
        return img[:, None]

    def heat2(self, g):
        """두 창 열지도 2채널: 채널 0 = 전체 창(W봉) 열지도, 채널 1 = 마지막 W/2 봉을 2배로 늘린 열지도(v2 교사의 60·120 두 창을 흉내)."""
        full = self.heat(g)                                     # (B,1,IMG_H,W*PX)
        half = full[..., full.shape[-1] // 2:]                  # 뒤쪽 절반 봉
        zoom = half.repeat_interleave(2, -1)[..., :full.shape[-1]]
        return torch.cat([full, zoom], 1)

    def heat(self, g):
        """8행 × W열 열지도(0~1): 행 = o/h/l/c 로그비(±10% → 0~1), 거래량/창평균(0~3), 30일 대비 거래량(±3σ), 시각, 요일. 행을 IMG_H 로, 열을 PX 로 늘린다."""
        x = self.sequence(g, True)                                   # (B,W,8)
        px = (x[..., :4] / 10.0).clamp(-1, 1) * 0.5 + 0.5; v = (x[..., 4:5] / 3.0).clamp(0, 1)
        img = torch.cat([px, v, x[..., 5:6] * 0.5 + 0.5, x[..., 6:8]], -1).transpose(1, 2)   # (B,8,W)
        img = img.repeat_interleave(IMG_H // 8, 1).repeat_interleave(PX, 2)
        return img[:, None]

    def render(self, g, volume=True, kind="bar", channels="gray", detrend=False):
        if kind == "heat":
            return self.heat(g)
        if kind == "heatx":
            return self.heatx(g)
        if kind == "heat2":
            return self.heat2(g)
        """→ (B,C,IMG_H,W*PX). gray: 1채널(JKX). multi: + 밀도(봉별 거래량/창 최대, 전체 높이 밝기) + 30일 대비 거래량 + 시각 + 요일 평면 = 5채널"""
        W = self.W
        ar = torch.arange(W, device=self.dev)
        win = self.window(g); o, h, l, c, v = win[..., :5].unbind(-1)
        if detrend:   # 창 안 로그가격의 최소제곱 추세를 나눠 순모멘텀을 지운다 — 형태·변동·거래량만 남김
            t = (ar - ar.float().mean())[None, :].float(); lc = torch.log(c.clamp_min(1e-12)); b = ((t * (lc - lc.mean(1, keepdim=True))).sum(1, keepdim=True) / (t * t).sum())
            tr = torch.exp(b * t); o, h, l, c = o / tr, h / tr, l / tr, c / tr
        ph = IMG_H - VOL_H - 2 if volume else IMG_H                   # 가격 패널 높이
        lo = l.min(1, keepdim=True).values; hi = h.max(1, keepdim=True).values
        scale = (ph - 1) / (hi - lo).clamp_min(1e-9)
        y = lambda p: ((hi - p) * scale).round().long().clamp(0, ph - 1)  # 위가 0
        rows = torch.arange(ph, device=self.dev)[None, :, None]         # (1,ph,1)
        img = torch.zeros(len(g), IMG_H, W * PX, device=self.dev)
        if kind == "line":
            yc = y(c); yp = torch.cat([yc[:, :1], yc[:, :-1]], 1)
            top = torch.minimum(yc, yp)[:, None, :]; bot = torch.maximum(yc, yp)[:, None, :]
            img[:, :ph, 1::PX] = ((rows >= top) & (rows <= bot)).float()
        else:
            yh, yl, yo, yc = y(h), y(l), y(o), y(c)
            if kind == "candle":
                top = torch.minimum(yo, yc)[:, None, :]; bot = torch.maximum(yo, yc)[:, None, :]
                body = ((rows >= top) & (rows <= bot)).float()
                wick = ((rows >= yh[:, None, :]) & (rows <= yl[:, None, :])).float()
                for j in range(PX):
                    img[:, :ph, j::PX] = body
                img[:, :ph, 1::PX] = torch.maximum(img[:, :ph, 1::PX], wick)
            else:  # JKX OHLC bar
                img[:, :ph, 1::PX] = ((rows >= yh[:, None, :]) & (rows <= yl[:, None, :])).float()
                bidx = torch.arange(len(g), device=self.dev)[:, None].expand(-1, W)
                cols = (ar * PX)[None, :].expand(len(g), -1)
                img[bidx, yo, cols] = 1.0
                img[bidx, yc, cols + 2] = 1.0
        if volume:
            vh = ((v / v.max(1, keepdim=True).values.clamp_min(1e-9)) * VOL_H).round().long()  # (B,W)
            vrows = torch.arange(VOL_H, device=self.dev)[None, :, None]
            img[:, IMG_H - VOL_H:, 1::PX] = (vrows >= (VOL_H - vh)[:, None, :]).float()
        if channels == "gray":
            return img[:, None]
        B_, Wp = len(g), W * PX
        dens = (v / v.max(1, keepdim=True).values.clamp_min(1e-9)).repeat_interleave(PX, 1)[:, None, :].expand(B_, IMG_H, Wp)   # 봉별 밀도를 열 밝기로
        vrel = win[..., 5].clamp(-3, 3).div(3).repeat_interleave(PX, 1)[:, None, :].expand(B_, IMG_H, Wp)                     # 30일 대비 거래량(창 밖 맥락)
        hour = win[:, -1, 6][:, None, None].expand(B_, IMG_H, Wp); dow = win[:, -1, 7][:, None, None].expand(B_, IMG_H, Wp)       # 판단 시각 평면
        planes = {"img": img, "dens": dens, "vrel": vrel, "hour": hour, "dow": dow}
        for j, nm in enumerate(MACRO_COLS):
            planes[nm] = win[:, -1, 8 + j].clamp(-2, 2)[:, None, None].expand(B_, IMG_H, Wp)
        return torch.stack([planes[k] for k in CHANNEL_SETS[channels]], 1)


# ---------- 모델 ----------
class SmallCNN(nn.Module):
    """JKX 형: (5×3 conv → BN → LeakyReLU → 2×1 maxpool) × 3 → 풀링 → FC. n_feat>0 이면 f1 융합."""

    def __init__(self, n_cls=3, n_feat=0, pool_w=1, in_ch=1):
        """pool_w=2 (i1f): 풀링을 2×2 로 해 폭도 줄인다 — 360px 폭 이미지에서 연산 약 8배 절감. in_ch: 다채널 이미지"""
        super().__init__()
        ch = [in_ch, 64, 128, 256]
        self.blocks = nn.Sequential(*[nn.Sequential(
            nn.Conv2d(ch[i], ch[i + 1], (5, 3), padding=(2, 1)), nn.BatchNorm2d(ch[i + 1]), nn.LeakyReLU(0.01), nn.MaxPool2d((2, pool_w)))
            for i in range(3)])
        self.pool = nn.AdaptiveAvgPool2d((6, 30))
        self.n_feat = n_feat
        if n_feat:
            self.img_fc = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(256 * 6 * 30, 128), nn.LeakyReLU(0.01))
            self.feat_fc = nn.Sequential(nn.Linear(n_feat, 64), nn.LeakyReLU(0.01))
            self.head = nn.Linear(128 + 64, n_cls)
        else:
            self.fc = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(256 * 6 * 30, n_cls))

    def forward(self, x, f=None):
        z = self.pool(self.blocks(x))
        if self.n_feat:
            return self.head(torch.cat([self.img_fc(z), self.feat_fc(f)], 1))
        return self.fc(z)


class ResNet18(nn.Module):
    def __init__(self, n_cls=3, depth=18, in_ch=1):
        super().__init__()
        import torchvision
        m = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1) if depth == 34 else torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        w = m.conv1.weight.sum(1, keepdim=True).repeat(1, in_ch, 1, 1) / in_ch
        m.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False); m.conv1.weight.data = w
        m.fc = nn.Linear(512, n_cls); self.m = m

    def forward(self, x, f=None):
        return self.m(Fn.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False))


class SeqTransformer(nn.Module):
    """j2: W×5 시퀀스 → 임베딩 64 + 위치 → 인코더 2층 → 평균 풀링 → FC"""

    def __init__(self, W, n_cls=3, d=64, in_dim=5):
        super().__init__()
        self.inp = nn.Linear(in_dim, d); self.pos = nn.Parameter(torch.zeros(1, W, d))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, d * 2, dropout=0.1, batch_first=True), 2)
        self.fc = nn.Linear(d, n_cls)

    def forward(self, x, f=None):
        return self.fc(self.enc(self.inp(x) + self.pos).mean(1))


def build_model(a, W):
    c = len(CHANNEL_SETS[a.channels])
    if a.render in ("heat", "heatx", "heatf"):
        c = 1
    if a.render == "heat2":
        c = 2
    return {"i1": lambda: SmallCNN(in_ch=c), "i1f": lambda: SmallCNN(pool_w=2, in_ch=c), "i2": lambda: ResNet18(in_ch=c), "i3": lambda: ResNet18(depth=34, in_ch=c), "j2": lambda: SeqTransformer(W, in_dim=8 if a.seq_ctx else 5),
            "f1": lambda: SmallCNN(n_feat=len(F.FEATURES), in_ch=c)}[a.model]()   # 속도 시험 결과 폭 보존(2×1 풀링)이 정확도에서 앞서 f1 도 i1 트렁크


# ---------- 학습 ----------
def inputs(wins, g, f, a):
    if a.model == "j2":
        return wins.sequence(g, a.seq_ctx), None
    if a.render == "heatf":
        return wins.heatf(g, f), f
    return wins.render(g, not a.no_volume, a.render, a.channels, a.detrend), f


@torch.no_grad()
def predict(model, wins, g, f, a, bs=512):
    model.eval(); out = []
    for i in range(0, len(g), bs):
        x, ff = inputs(wins, g[i:i + bs], None if f is None else f[i:i + bs], a)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(x, ff).float()
        if getattr(a, "teacher", ""):
            sc = o[:, 1] - o[:, 2]; out.append(torch.stack([torch.zeros_like(sc), sc.clamp(min=0), (-sc).clamp(min=0)], 1).cpu())
        else:
            out.append(torch.softmax(o, 1).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True); ap.add_argument("--W", type=int, required=True); ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--k", type=float, default=2.0); ap.add_argument("--label", default="ksigma", choices=["ksigma", "fixed", "binary", "rel", "relbin"]); ap.add_argument("--stride", type=int, default=0, help="창 간격(봉), rel 라벨·앙상블 정렬용"); ap.add_argument("--fixed", type=float, default=0.02)
    ap.add_argument("--model", default="i1", choices=["i1", "i1f", "i2", "i3", "j2", "f1"])
    ap.add_argument("--top", type=int, default=200); ap.add_argument("--bases")
    ap.add_argument("--universe", default="", help="S4 종목군 G1~G5 (평가는 전체 홀드아웃)"); ap.add_argument("--delisted", default="in", choices=["in", "out"])
    ap.add_argument("--val", default="2025-09"); ap.add_argument("--test", default="2025-12"); ap.add_argument("--test_end", default="2026-03")
    ap.add_argument("--start", default="2023-01"); ap.add_argument("--train_months", type=int, default=0, help="IS 고정 길이(개월). 0 이면 확장형")
    ap.add_argument("--cap", type=int, default=2_000_000); ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_volume", action="store_true"); ap.add_argument("--shuffle", action="store_true"); ap.add_argument("--render", default="bar"); ap.add_argument("--detrend", action="store_true"); ap.add_argument("--seq_ctx", action="store_true")
    ap.add_argument("--channels", default="gray", choices=list(CHANNEL_SETS), help="gray 1채널(JKX) · c2 차트+밀도 · c3 +시각 · multi 5채널(+30일 거래량·요일)")
    ap.add_argument("--px", type=int, default=3, help="봉당 픽셀 폭"); ap.add_argument("--img_h", type=int, default=96, help="이미지 높이")
    ap.add_argument("--out", required=True); ap.add_argument("--save_pred", action="store_true")
    ap.add_argument("--teacher", default="", help="증류: 교사 점수 npz(ts, base, score). 학생은 이미지에서 교사 점수를 회귀(MSE)로 배운다"); ap.add_argument("--teacher_xs", action="store_true", help="교사 점수를 판단 시각별로 z-정규화(순수 횡단면 목표)"); ap.add_argument("--tail_w", type=float, default=0.0, help="증류 손실 꼬리 가중: |교사 z|>1 인 행에 1+K 배(십분위 양끝을 더 정확히)")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    global PX, IMG_H; PX, IMG_H = a.px, a.img_h        # 이미지 기하 변형(사장 파라미터 탐색)
    torch.backends.cudnn.benchmark = True      # 입력 크기가 고정이라 커널 자동 선택이 이득
    dev = "cuda"
    u = pd.read_csv("work/universe.csv"); u = u[u.exclude.fillna("") == ""]
    if a.universe:
        from hyfe.pilot_gbm import universe_bases
        holdout = set(u[u.holdout].base)
        bases = sorted(set(universe_bases(u, a.universe, a.delisted)) | holdout)
    else:
        holdout = set(u[u.holdout].base)
        if a.bases:   # 종목 목록: @파일(한 줄에 하나, 폴드별 유동성 상위 = GBM 과 같은 유니버스) 또는 쉼표 구분
            bases = [l.strip() for l in open(a.bases[1:]) if l.strip()] if a.bases.startswith("@") else a.bases.split(",")
        else:
            u = u[u.liq_rank <= a.top]; bases = u.base.tolist()
    t0 = time.time()
    wins = Windows(bases, a.res, a.W, a.H, a.k, dev, a.start, a.test_end, a.label, a.fixed, need_feats=(a.model == "f1" or a.render == "heatf"), stride=a.stride or None, macro=a.channels.endswith("m"))
    ix = wins.idx
    if a.render == "heatf":
        from hyfe.pilot_gbm import add_xs
        ix = add_xs(ix); wins.idx = ix
    if a.teacher:   # 교사(피처 GBM 의 표본 밖 점수)를 (ts, base) 로 붙인다. 없는 행은 학습에서 제외
        tz = np.load(a.teacher, allow_pickle=True); tdf = pd.DataFrame({"ts": tz["ts"], "base": tz["base"].astype(str), "tscore": tz["score"].astype(np.float32)})
        if a.teacher_xs:
            g_ = tdf.groupby("ts").tscore; tdf["tscore"] = ((tdf.tscore - g_.transform("mean")) / (g_.transform("std") + 1e-9)).astype(np.float32)
        ix = ix.merge(tdf, on=["ts", "base"], how="left"); wins.idx = ix
        print(f"teacher rows {tdf.shape[0]} matched {int(ix.tscore.notna().sum())}/{len(ix)}", flush=True)
    emb = (a.W + a.H) * B.RES_MIN[a.res] * 60_000
    seen = ~ix.base.isin(holdout)
    is_from = ms(a.val) - a.train_months * 30 * 86_400_000 if a.train_months else 0
    tr = ix[seen & (ix.ts < ms(a.val) - emb) & (ix.ts >= is_from)]          # IS
    if a.teacher:
        tr = tr[tr.tscore.notna()]
    va = ix[seen & (ix.ts >= ms(a.val)) & (ix.ts < ms(a.test) - emb)]
    te = ix[(ix.ts >= ms(a.test)) & (ix.ts < ms(a.test_end))]
    if len(tr) > a.cap:
        tr = tr.sample(a.cap, random_state=a.seed)
    y_tr = tr.label.to_numpy().copy()
    if a.shuffle:
        np.random.shuffle(y_tr)
    print(f"train {len(tr)} val {len(va)} test {len(te)} (unseen {te.base.isin(holdout).sum()}) load {time.time()-t0:.0f}s", flush=True)

    feat = None
    if a.model == "f1" or a.render == "heatf":   # 표준화 통계는 학습 집합에서만. heatf = GBM 이 보는 피처 27 + 횡단면 순위 10 을 이미지 행으로
        FEATS = list(F.FEATURES) + (XS_FEATS if a.render == "heatf" else [])
        mu, sd = tr[FEATS].mean(), tr[FEATS].std() + 1e-9
        feat = lambda d: torch.tensor(((d[FEATS] - mu) / sd).to_numpy(np.float32), device=dev)
    f_tr, f_va, f_te = (feat(tr), feat(va), feat(te)) if feat else (None, None, None)

    model = build_model(a, a.W).to(dev)
    lr = a.lr or {"i1": 1e-3, "i1f": 1e-3, "i2": 1e-4, "i3": 1e-4, "j2": 5e-4, "f1": 1e-3}[a.model]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    g_tr = torch.tensor(tr.g.to_numpy(), device=dev); y_trt = torch.tensor(y_tr, device=dev)
    t_trt = torch.tensor(tr.tscore.to_numpy(np.float32), device=dev) if a.teacher else None
    t_va = va.tscore.to_numpy(np.float32) if a.teacher else None
    g_va = torch.tensor(va.g.to_numpy(), device=dev); g_te = torch.tensor(te.g.to_numpy(), device=dev)
    best, best_state, bad, hist = -1, None, 0, []
    for ep in range(a.epochs):
        model.train(); perm = torch.randperm(len(g_tr), device=dev); tl = 0; te0 = time.time()
        for i in range(0, len(perm), a.bs):
            j = perm[i:i + a.bs]
            x, ff = inputs(wins, g_tr[j], None if f_tr is None else f_tr[j], a)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(x, ff).float()
                if a.teacher and a.tail_w > 0:
                    w_ = 1.0 + a.tail_w * (t_trt[j].abs() > 1.0).float(); loss = (w_ * (o[:, 1] - o[:, 2] - t_trt[j]) ** 2).sum() / w_.sum()
                else:
                    loss = Fn.mse_loss(o[:, 1] - o[:, 2], t_trt[j]) if a.teacher else Fn.cross_entropy(o, y_trt[j])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tl += loss.item() * len(j)
        pv = predict(model, wins, g_va, f_va, a); r = M.evaluate(va.label.to_numpy(), pv, va.fwd.to_numpy(), va.sigH.to_numpy())
        if a.teacher:   # 조기종료 = 검증 구간에서 교사 점수와의 상관(높을수록 좋음)
            m_ = ~np.isnan(t_va); sv = pv[:, 1] - pv[:, 2]; r["ap"] = float(np.corrcoef(sv[m_], t_va[m_])[0, 1]) if m_.sum() > 100 else 0.0
        hist.append({"epoch": ep, "loss": tl / len(perm), "val_ap": r["ap"], "val_spread_z": r.get("spread_z"), "sec": round(time.time() - te0)})
        print(f"ep{ep} loss {tl/len(perm):.4f} val_{'corr' if a.teacher else 'ap'} {r['ap']:.4f} spread_z {r.get('spread_z', float('nan')):.3f} {time.time()-te0:.0f}s", flush=True)
        if r["ap"] > best:
            best, bad = r["ap"], 0; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 2:
                break
    model.load_state_dict(best_state)
    pv = predict(model, wins, g_va, f_va, a); pt = predict(model, wins, g_te, f_te, a)
    un = te.base.isin(holdout).to_numpy()
    ev = lambda d, p: M.evaluate(d.label.to_numpy(), p, d.fwd.to_numpy(), d.sigH.to_numpy())
    res = {"args": vars(a), "n_train": len(tr), "n_val": len(va), "n_test": len(te), "n_unseen": int(un.sum()), "hist": hist,
           "feat_cols": (FEATS if feat else None), "feat_mu": (mu.tolist() if feat else None), "feat_sd": (sd.tolist() if feat else None),   # 실전 추론용 표준화 통계
           "val": ev(va, pv), "test": ev(te[~un], pt[~un]), "unseen": ev(te[un], pt[un]) if un.sum() > 500 else None,
           "sec": round(time.time() - t0)}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    if a.save_pred:
        np.savez_compressed(a.out.replace(".json", "_pred.npz"), ts=te.ts.to_numpy(), base=te.base.to_numpy().astype(str), y=te.label.to_numpy(),
                            p=pt, fwd=te.fwd.to_numpy(), sigH=te.sigH.to_numpy())
        torch.save(best_state, a.out.replace(".json", ".pt"))
    print("DONE", json.dumps({k: res[k] for k in ("val", "test", "unseen")}), flush=True)


if __name__ == "__main__":
    main()
