# 근거 논문 3편

읽는 순서대로 나열한다. 요약 과제(팀원 A)는 이 세 편이 대상이다.

## 1. (Re-)Imag(in)ing Price Trends — Jiang, Kelly, Xiu

- 파일: `jkx_imaging_price_trends.pdf` (공개 워킹페이퍼판, NUS AIDF 미러)
- 출판본: Journal of Finance 78(6), 2023 · SSRN <https://ssrn.com/abstract=3756587>
- 차트를 이미지로 바꿔 CNN에 학습시키는 접근의 원조다. 미국 주식에서 이미지 기반 신호가
  기존 추세 지표보다 낫다는 걸 보였다. 우리 이미지 표현(캔들+거래량 그레이스케일)이
  이 논문 방식을 따른다.

## 2. Visualizing Price Trends in China: A Multi-Channel Grayscale CNN Approach

- SSRN <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5136032> (2025-07 게재)
- ⚠ PDF가 이 폴더에 없다. SSRN이 자동 다운로드를 막아서 브라우저로 직접 받아야 한다
  (무료, 회원가입만 필요). 받으면 이 폴더에 올려둘 것 — A1 과제의 첫 단추다.
- JKX 방식을 중국 A주에 재현한 연구다. 미국 밖에서도 통한다는 근거가 된다.

## 3. Visual Chart Representations for Cryptocurrency Regime Prediction

- 파일: `crypto_chart_finetuning_arxiv2605.00875.pdf`
- arXiv <https://arxiv.org/abs/2605.00875> (2026)
- 암호화폐 차트에 사전학습 비전 모델을 파인튜닝하면 성능이 4~16% 오른다는 실험.
  다만 BTC·ETH의 추세 분류에서 멈췄다 — 천 개 알트코인의 급등락 조기 예측은 비어 있고,
  우리가 노리는 자리가 바로 거기다.

## 요약 양식 (A1 과제)

논문마다 1쪽, 세 칸이면 된다.

1. 무엇을 했나 — 데이터·대상·목표
2. 어떻게 했나 — 이미지 표현·모델·평가 방식
3. 우리 연구에 가져올 점 — 그대로 쓸 것 / 다르게 할 것
