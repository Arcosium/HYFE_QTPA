# 크립토 무기한선물 1분봉 데이터셋

암호화폐 **무기한선물(perpetual futures)** 의 1분봉 OHLCV 다.
2023-01-01부터 현재까지, **상장폐지된 종목을 포함해** 기간 중 한 번이라도 거래된 전 종목을 담았다.

| | |
|---|---|
| 해상도 | 1분봉 (시가·고가·저가·종가·거래량) |
| 기간 | 2023-01-01 ~ 현재 (이후 자동으로 계속 쌓임) |
| 종목 | **1,060개** — binance 832 · bybit 80 · bybit 상장폐지분 148 |
| 시간대 | **전부 UTC** (한국시간 아님) |
| 형식 | Parquet, 종목당 1파일 |

---

## 1. 왜 상장폐지 종목이 들어 있나 — 생존편향

지금 상장돼 있는 종목만 모으면 **망해서 사라진 코인이 전부 빠진다.**
살아남은 것만 놓고 "코인 수익률이 이렇다"고 하면 그 숫자는 위로 부풀려진 거짓이다.
이 편향을 생존편향(survivorship bias)이라 하고, 백테스트를 실제보다 좋아 보이게 만드는 가장 흔한 원인이다.

그래서 이 데이터셋의 모집단은 '현재 상장 종목'이 아니라 **'기간 중 한 번이라도 거래된 종목'** 이다.

- binance: 벌크 덤프에 상장폐지분 파일이 그대로 남아 있어 전부 회수 (전체 986심볼 중 **413개가 이미 상장폐지**)
- bybit: kline API 가 상장폐지 심볼에 **빈 응답**을 준다 → 체결(tick) 덤프를 받아 1분봉으로 직접 집계
- 예시로 회수된 죽은 종목: `10000ELON` `10000LADYS` `1000000CHEEMS` `10000NFT` `10000STARL` `1000IQ50` …

`metadata/coverage.csv` 의 `delisted` 열로 구분할 수 있다.

## 2. 컬럼

| 컬럼 | 뜻 |
|---|---|
| `ts` | 봉 **시작** 시각. UTC epoch 밀리초 |
| `datetime` | 같은 값을 사람이 읽는 UTC 문자열로 |
| `venue` | 데이터를 받은 거래소 (`binance` / `bybit`) |
| `base` | 코인 티커 (`BTC`, `DOGE` …). 파일명과 같다 |
| `symbol` | 거래소 원본 심볼 (`BTCUSDT`, `BTCPERP` …) |
| `open` `high` `low` `close` | 시가 · 고가 · 저가 · 종가 |
| `volume` | 거래량 (**코인 수량** 기준) |
| `quote_volume` | 거래대금 (**달러** 기준). 거래소가 안 주면 빈값 |

## 3. 출처 — 종목마다 만들어진 방법이 다르다

`metadata/coverage.csv` 의 `source` 열을 반드시 확인하라. 셋의 성격이 다르다.

| source | 종목 | 방법 | 신뢰도 |
|---|---|---|---|
| `binance` | 832 | 거래소 공식 1분봉 벌크 덤프 | 거래소가 확정한 봉 그대로 |
| `bybit` | 80 | 거래소 공식 kline API | 거래소가 확정한 봉 그대로 |
| `bybit_trades` | 148 | **체결 데이터를 직접 1분봉으로 집계** | ⚠ 거래소 공식 봉이 아니다 |

`bybit_trades` 는 상장폐지 때문에 공식 봉을 못 구해 체결 하나하나를 모아 재구성한 것이다.
집계 규칙은 분 단위로 자른 뒤 첫 체결가=시가, 최고=고가, 최저=저가, 마지막=종가, 수량 합=거래량이다.
거래소 공식 봉과 소수점 끝자리가 다를 수 있으니, **정밀도가 중요한 분석에서는 이 148종목을 따로 취급**하라.

## 4. 알아둘 한계

- **거래가 없던 분에는 봉이 없다.** 시계열을 균일 간격으로 만들려면 직접 채워야 한다(보통 종가 forward-fill).
  거래가 활발한 종목은 결측이 사실상 없고, 죽어가는 종목일수록 구멍이 커진다.
- **종목당 거래소 1곳만 담았다.** 같은 코인을 여러 거래소에서 중복 수집하지 않았다 —
  가격은 사실상 같고 용량만 몇 배가 되기 때문. 따라서 **거래소간 스프레드·차익거래 분석은 이 데이터로 못 한다.**
- **`quote_volume` 이 빈 종목이 있다.** 거래소가 안 주는 경우이며, `close × volume` 으로 지어내지 않았다.
- **종목마다 시작일이 다르다.** 2023-01-01이 아니라 그 코인의 상장일부터다. `coverage.csv` 의 `first_utc` 참조.
- **코인이 아닌 종목이 섞여 있다.** 거래소가 상장한 토큰화 주식·원자재다 —
  `UBER` `AMD` `DIS` `NOK` `EWZ` `NVO` `SPACEX` `OPENAI` `ANTHROPIC` `NATGAS` `XPT`(백금) `COPPER` `URNM` 등.
  순수 코인만 필요하면 걸러내라.
- **달러 계열(USDT·USDC·BUSD·USD) 마켓만 담았다.** `ETHBTC` 같은 코인 마켓 선물은 뺐다 —
  거래대금 단위가 달러가 아니라 다른 종목과 섞으면 그 종목이 통째로 왜곡된다.
- **거래소 4곳(grvt·hyperliquid·paradex·backpack)에만 있는 종목은 과거가 없다.**
  이 거래소들은 과거 조회 상한이 하루~며칠이라 이력을 안 준다. 해당 종목은 수집 시작 시점 이후만 존재한다.

## 5. 읽는 법

```python
import pandas as pd
df = pd.read_parquet("parquet/BTC.parquet")

# 전 종목을 한 번에 (duckdb 권장 — 메모리에 다 안 올리고 처리한다)
import duckdb
duckdb.sql("""
  SELECT base, date_trunc('day', datetime) d, last(close ORDER BY ts) px
  FROM read_parquet('parquet/*.parquet') GROUP BY 1,2
""").df()
```

R 은 `arrow::read_parquet()`, polars 는 `pl.read_parquet()` 로 똑같이 읽힌다.

## 6. 같이 든 파일

| 파일 | 내용 |
|---|---|
| `parquet/<종목>.parquet` | 봉 데이터 본체 |
| `metadata/coverage.csv` | **종목별 행수·시작일·종료일·출처·상장폐지 여부** — 먼저 볼 것 |
| `metadata/universe.csv` | 종목 → 거래소·심볼 배정 |
| `metadata/README.csv` | 한 장짜리 요약 |
| `metadata/DATASET.md` | 이 문서 |
