# 데이터 안내

연구에 쓰는 데이터는 자체 수집한 CryptoBars 1분봉이다. 전체는 8.3억 행 · 1,057종목 ·
2023-01-01~현재 · 6개 거래소(binance·bybit·backpack·paradex·grvt·hyperliquid)이고
상장폐지 코인까지 담아 생존편향이 없다. 상세 명세는 `DATASET.md`에 있다.

## 이 폴더에 있는 것

| 파일 | 내용 |
|------|------|
| `DATASET.md` | 데이터셋 정식 명세서 (컬럼·수집 방식·주의사항) |
| `coverage.csv` | 종목별 행수·기간·상폐 여부 — C1 과제의 출발점 |
| `universe.csv` | 종목 목록 |
| `sample/` | 연습용 샘플: 6종목 × 2026-07~08 두 달치 1분봉 |

## 샘플 데이터

`sample/base=<종목>/part-<연-월>.parquet` 구조다. 종목은 BTC·ETH·SOL·DOGE·AGIX·WAVES.
컬럼: `ts`(ms, UTC) · `venue`(거래소) · `base` · `symbol` · `open` · `high` · `low` ·
`close` · `volume` · `quote_volume`.

```python
import pandas as pd, glob
df = pd.concat(pd.read_parquet(p) for p in glob.glob("sample/base=BTC/*.parquet"))
df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
```

## 전체 데이터 (20GB)

깃허브에 올리기엔 커서 구글드라이브 `CryptoBars` 폴더에 있다(parquet 1,057개).
링크는 팀장이 회합 때 공유한다. 과제 대부분은 팀장이 주는 정제 샘플로 충분하니
전체를 받을 일은 거의 없다.

## 다룰 때 조심할 것

- hyperliquid·paradex는 `quote_volume`이 비어 있다.
- bybit 145종목은 거래소 공식 봉이 아니라 체결 기록을 우리가 직접 분으로 묶은 것이다.
  정밀 분석에서는 따로 표시한다.
- 코인이 아닌 것(토큰화 주식·원자재: UBER·AMD·NATGAS 등)이 섞여 있다 — 제외 목록 대상.
- "파일이 있다"와 "데이터가 다 있다"는 다르다. 종목×월 단위로 빈 구간을 검산해야 한다.
