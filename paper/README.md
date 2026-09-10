# paper/ — 논문 조립 자료

완성 원고(docx·PDF)는 저장소에 두지 않고 팀장이 회합에서 공유한다. 여기에는 원고를 다시 조립할 수 있는 것만 둔다.

| 파일 | 내용 |
|---|---|
| `build_docx.py` | 학술제 양식 docx 조립 스크립트. 장·절·표·그림 순서와 본문 문장이 전부 이 파일에 있다(표지 이름은 비워 둠). `OUTDIR` 경로는 팀장 서버 기준이라 그대로 돌리려면 바꿔야 한다. |
| `fig.py` | 그림 1~19 생성(하우스 규칙: ink + 액센트 1 + 회색 2단, Noto Sans KR). 저장소 `results/` 를 읽는다. |
| `mlp_eval.py` | MLP 대조군 표(`mlp_table.csv`) 계산 |
| `figs/` | 논문에 들어간 그림 PNG 20장(그림 14 = AutoCrypto 화면 캡처) |
| `candle_settings_table.csv` | 가설 0 캔들 설정 25개와 폴드별 spread_z·적중률(표 4·7) |
| `btc_compare.csv` | 롱숏 vs BTC 보유 구간별 수익·MDD·변동성(표 20) |
| `mlp_table.csv` · `funding_table.csv` · `liquidity_legs.csv` | MLP 대조군, 펀딩비 반영, 다리별 유동성 표 |

확장 실험 표(가림·배치 재학습·지평·용량·국면·직교성·숏 다리)는 `../results/ext_*.csv`, `occlusion_table_*.csv`, `stack_table.csv`, `dsr_table.csv` 에 있고 과정은 `../실험일지.md` 9/10 항목이다.
