# -*- coding: utf-8 -*-
"""HYFE_QTPA 논문 docx 조립 — 학술제 양식(표지·연구 초록·주요어·제N장/제N절·A4·바탕 10.5pt·가운데 쪽번호). 출력은 vault 에만."""
import os, sys, json, glob, re
import pandas as pd
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, WD_BREAK
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

os.chdir("/home/arcosium/projects/HYFE_QTPA"); sys.path.insert(0, ".")
OUTDIR = "/home/arcosium/vault/HYFE_QTPA/paper"; FIG = f"{OUTDIR}/figs"; OUT = f"{OUTDIR}/HYFE_QTPA_논문_v1.docx"
EA, LAT = "바탕", "Times New Roman"

doc = Document()
sec = doc.sections[0]; sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.5); sec.top_margin = Cm(2.5); sec.bottom_margin = Cm(2.2)

def set_font(run, size=10.5, bold=None, ea=EA, lat=LAT, color=None):
    run.font.size = Pt(size); run.font.name = lat
    rpr = run._element.get_or_add_rPr(); rf = rpr.find(qn("w:rFonts"))
    if rf is None: rf = OxmlElement("w:rFonts"); rpr.append(rf)
    rf.set(qn("w:ascii"), lat); rf.set(qn("w:hAnsi"), lat); rf.set(qn("w:eastAsia"), ea)
    if bold is not None: run.bold = bold
    if color: run.font.color.rgb = RGBColor(*color)

st = doc.styles["Normal"]; st.font.size = Pt(10.5); st.font.name = LAT
st.element.rPr.rFonts.set(qn("w:eastAsia"), EA)
pf = st.paragraph_format; pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE; pf.line_spacing = 1.55; pf.space_after = Pt(0)

def para(text="", size=10.5, bold=False, align="justify", indent=True, before=0, after=0, ea=EA, color=None, italic=False):
    p = doc.add_paragraph(); p.alignment = {"justify": WD_ALIGN_PARAGRAPH.JUSTIFY, "center": WD_ALIGN_PARAGRAPH.CENTER, "left": WD_ALIGN_PARAGRAPH.LEFT, "right": WD_ALIGN_PARAGRAPH.RIGHT}[align]
    p.paragraph_format.space_before = Pt(before); p.paragraph_format.space_after = Pt(after)
    if indent and align == "justify": p.paragraph_format.first_line_indent = Cm(0.4)
    if text:
        r = p.add_run(text); set_font(r, size, bold, ea=ea, color=color); r.italic = italic
    return p

def chapter(text):
    p = doc.add_paragraph(); p.paragraph_format.page_break_before = True; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(18); p.paragraph_format.space_after = Pt(18); set_font(p.add_run(text), 16, True, ea="맑은 고딕", lat="Arial")

def section(text):
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(14); p.paragraph_format.space_after = Pt(6); p.paragraph_format.keep_with_next = True
    set_font(p.add_run(text), 12, True, ea="맑은 고딕", lat="Arial")

def sub(text):
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(8); p.paragraph_format.space_after = Pt(3); p.paragraph_format.keep_with_next = True
    set_font(p.add_run(text), 10.5, True, ea="맑은 고딕", lat="Arial")

def caption(text, before=2, after=6):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before = Pt(before); p.paragraph_format.space_after = Pt(after); p.paragraph_format.keep_with_next = True
    set_font(p.add_run(text), 9.5, True, ea="맑은 고딕", lat="Arial")

def shade(cell, hexfill):
    tcPr = cell._element.get_or_add_tcPr(); shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hexfill); tcPr.append(shd)

def borders(table):
    tbl = table._element; tblPr = tbl.tblPr; b = OxmlElement("w:tblBorders")
    for edge, sz in [("top", 8), ("bottom", 8), ("insideH", 4)]:
        e = OxmlElement(f"w:{edge}"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), str(sz)); e.set(qn("w:color"), "444444"); b.append(e)
    for edge in ["left", "right", "insideV"]:
        e = OxmlElement(f"w:{edge}"); e.set(qn("w:val"), "nil"); b.append(e)
    tblPr.append(b)

def table(cap, header, rows, widths=None, size=8.5, note=None, align_first_left=True):
    cap, key = (cap.split("@@") + [None])[:2]
    caption(f"<표 {table.n}> {cap}")
    if key: table.keys[key] = table.n
    table.n += 1
    t = doc.add_table(rows=1 + len(rows), cols=len(header)); t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.autofit = widths is None
    borders(t)
    for j, h in enumerate(header):
        c = t.rows[0].cells[j]; c.text = ""; p = c.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER; set_font(p.add_run(str(h)), size, True, ea="맑은 고딕", lat="Arial"); shade(c, "F2F2F2")
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = t.rows[i + 1].cells[j]; c.text = ""; p = c.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.LEFT if (j == 0 and align_first_left) else WD_ALIGN_PARAGRAPH.CENTER
            bold = isinstance(v, tuple); txt = v[0] if bold else v; set_font(p.add_run(str(txt)), size, bold if bold else None, ea=EA, lat=LAT)
            p.paragraph_format.line_spacing = 1.15
    if widths:
        for row in t.rows:
            for j, w in enumerate(widths): row.cells[j].width = Cm(w)
        for gc, w in zip(t._tbl.tblGrid.findall(qn("w:gridCol")), widths): gc.set(qn("w:w"), str(int(Cm(w).twips)))
    for row in t.rows[:-1]:   # 표가 쪽 사이에서 갈라지지 않게(마지막 행 빼고 다음 행과 붙임)
        trPr = row._tr.get_or_add_trPr(); cs = OxmlElement("w:cantSplit"); trPr.append(cs)
        for c in row.cells:
            for p in c.paragraphs: p.paragraph_format.keep_with_next = True
    if note: para(note, 8.5, align="left", indent=False, before=3, after=8, color=(90, 90, 90))
    else: para("", after=6)
table.n = 1; table.keys = {}

def figure(name, cap, width=15.0):
    cap, key = (cap.split("@@") + [None])[:2]
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before = Pt(6); p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(f"{FIG}/{name}" if not name.startswith("/") else name, width=Cm(width))
    caption(f"<그림 {figure.n}> {cap}", before=2, after=10)
    if key: figure.keys[key] = figure.n
    figure.n += 1
figure.n = 1; figure.keys = {}

def bullet(text, size=10.5):
    p = doc.add_paragraph(); p.paragraph_format.left_indent = Cm(0.6); p.paragraph_format.first_line_indent = Cm(-0.4); p.paragraph_format.space_after = Pt(2)
    set_font(p.add_run("• " + text), size)

def add_page_number_footer():
    footer = sec.footer; p = footer.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(); fld = OxmlElement("w:fldSimple"); fld.set(qn("w:instr"), "PAGE"); rr = OxmlElement("w:r"); t = OxmlElement("w:t"); t.text = "1"; rr.append(t); fld.append(rr); r._element.append(fld); set_font(r, 9)

# ---------------- 데이터 사실 ----------------
cov = pd.read_csv("data/coverage.csv"); use = set(l.strip() for l in open("work/all_usable.txt") if l.strip()); u = cov[cov.base.isin(use)]
N_ALL, ROWS_ALL, DEL_ALL = len(cov), int(cov.rows.sum()), int(cov.delisted.sum()); N_USE, ROWS_USE, DEL_USE = len(u), int(u.rows.sum()), int(u.delisted.sum())
SRC_USE = u.source.value_counts().to_dict(); SRC_ALL = cov.source.value_counts().to_dict()
seed = pd.read_csv("results/seed_table.csv"); ens = pd.read_csv("results/ensemble10_table.csv")
def fmt(x, d=2): return "—" if pd.isna(x) else ("< 0.0001" if (d >= 3 and x < 0.0001) else f"{x:.{d}f}")

# ================= 표지 =================
for _ in range(6): para("", after=0)
para("캔들 차트의 실패에서 CNN 이 읽는 열지도까지,", 20, True, "center", indent=False, after=2)
para("암호화폐 횡단면 방향 예측", 20, True, "center", indent=False, after=8)
para("- 차트 이미지 설계와 소형 CNN 의 알파 -", 13, False, "center", indent=False, after=40)
para("2026년 9월 10일 (초안 v1)", 12, False, "center", indent=False, after=36)
para("한양대학교", 13, False, "center", indent=False, after=2); para("파이낸스경영학과", 13, False, "center", indent=False, after=2); para("학술제 (   )팀", 13, False, "center", indent=False, after=36)
for role, name in [("팀   장", "          "), ("팀   원", "          "), ("팀   원", "          "), ("팀   원", "          ")]:
    para(f"{role}      {name}      (인)", 12, False, "center", indent=False, after=6)

# ================= 연구 초록 =================
p = doc.add_paragraph(); p.paragraph_format.page_break_before = True; p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after = Pt(16); set_font(p.add_run("연 구 초 록"), 16, True, ea="맑은 고딕", lat="Arial")
ABS = [
 "차트를 이미지로 바꿔 CNN 에 학습시키는 접근은 미국 주식(Jiang·Kelly·Xiu 2023)과 중국 A주에서 성과를 보였지만 암호화폐에서는 BTC·ETH 두 종목의 추세 분류에 머물러 있었다. 이 연구는 상장폐지 종목을 포함한 무기한선물 740종목의 1분봉(2023-01~2026-05, 7.7억 행)을 4시간봉으로 정리하고 같은 판단 시각의 유니버스 평균을 뺀 14일 상대수익 부호를 라벨로 삼아, 주식에서 성과를 낸 캔들·거래량 차트 이미지가 암호화폐 횡단면에서도 유효하다는 가설(가설 0)을 검정하였다.",
 "실험 결과 가설 0 은 사전 등록 기준에 미달하여 기각되었다. 격자·창 길이·채널·거래량 유무·추세 제거·해상도·종목 수·학습률·유니버스를 바꾼 25개 설정 100개 폴드에서 캔들 이미지 CNN 의 시험 구간 상대방향 점수 차는 −0.22~+0.16, 평균 −0.01 로 0 근처에 머물렀고 사전 등록 기준(spread_z 0.2 이상, 4폴드 중 3폴드)에 든 설정은 하나도 없었다. 기각의 원인은 두 가지로 확인되었다. 첫째, 모델은 차트의 형태가 아니라 창 안 모멘텀과 수익률 왜도를 학습하였고 그 예측 부호는 분기마다 반전되었다. 둘째, 창 안에서 정규화되는 캔들 이미지에는 해당 종목의 유니버스 안 위치와 거래량의 장기 맥락 정보가 소실되어 있었다.",
 "이에 정규화 기준을 창 밖에 둔 수치 열지도 이미지(heatf)를 설계하였다. OHLC 로그비·거래량·30일 대비 거래량·시각·요일 8행, 같은 시각 유니버스 안 순위 3행, 창 끝 요약 피처 37행을 96×180 그레이스케일로 그린다. 순위 3행만 추가해도 부호가 양수로 전환되었고 피처 행까지 추가하자 75만 파라미터 소형 CNN 이 교사 없이 같은 데이터·같은 라벨·같은 평가의 LightGBM 을 넘어섰다. 사전 등록한 시드 0~9 전부의 z-평균 앙상블은 롤링 4폴드 코호트 롱숏(상·하위 10%, 14일 보유, 왕복 0.2%)에서 폴드별 Sharpe 5.59·4.95·2.92·2.28, 합산 408일 Sharpe 3.53(Newey-West p < 0.0001, MDD −4.2%)이었고 LightGBM 은 같은 구간 2.18 이었다. 설정 확정 뒤 한 번만 쓴 홀드아웃 2026-03~05 에서 앙상블 Sharpe 는 2.49(p 0.019)였으며 시드 열 개 모두 1.3 이상이었다. 점수를 종목 간 무작위로 섞은 대조는 모든 구간에서 음수였다. 실측 펀딩비를 반영해도 합산 2.95(p 0.0005), 홀드아웃 2.00(p 0.036)으로 유의성이 유지된다. 여기에 방향은 CNN 이 정하고 다리 안 비중을 급등·급락 발생 확률로 주는 2단계 결합(가설 3)은 합산 Sharpe 를 3.87 로 올렸다(Δ +0.33, 단측 p 0.04, 4폴드 중 3폴드 개선). 행군을 가리거나 빼고 다시 학습한 실험은 알파의 대부분이 창 밖에서 정규화된 맥락 행에서 나오고 봉의 국소 순서는 쓰이지 않으며 시계열 열지도가 작지만 0 이 아닌 몫을 더함을 보였다. 성과를 가르는 요인은 CNN 의 채택 여부가 아니라 CNN 에 어떤 이미지를 제시하는가라는 것이 본 연구의 결론이다.",
]
for a in ABS: para(a, 10.5, after=6)
para("", after=14); para("주요어 : 차트 이미지, CNN, 암호화폐, 횡단면 방향 예측, 열지도 표현, 시드 앙상블, 생존편향, 코호트 롱숏", 10.5, True, "left", indent=False)

# ================= 목차 =================
p = doc.add_paragraph(); p.paragraph_format.page_break_before = True; p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after = Pt(14); set_font(p.add_run("목    차"), 16, True, ea="맑은 고딕", lat="Arial")
TOC = ["제1장 서 론", "  제1절 연구 배경과 선행연구", "  제2절 가설 0: 캔들 차트 이미지의 스물다섯 설정", "  제3절 이 연구의 차이, 기여와 구성",
       "제2장 공통 잣대: 데이터, 라벨, 평가 규약", "  제1절 데이터와 생존편향 실측", "  제2절 라벨과 지평", "  제3절 폴드와 홀드아웃", "  제4절 코호트 롱숏과 판정 지표", "  제5절 대조군 LightGBM", "  제6절 왜 LightGBM 이고 왜 소형 CNN 인가",
       "제3장 실패의 해부: 캔들 차트는 왜 실패했는가", "  제1절 스물다섯 설정의 시험 결과", "  제2절 원인 1: 모멘텀과 왜도의 암기", "  제3절 원인 2: 창 안 정규화가 지운 맥락", "  제4절 이미지는 구조화 모델에 보태지 않았다",
       "제4장 표현의 재설계: CNN 이 읽을 수 있는 열지도(heatf)", "  제1절 원인 둘을 뒤집는 가설 1", "  제2절 heatf 의 구성", "  제3절 단계별 제거 실험", "  제4절 사전학습 백본과 증류", "  제5절 이미지가 본질인가: MLP 대조군",
       "제5장 채택과 검증: 교사 없는 소형 CNN 과 3중 검증", "  제1절 가설 2 와 모델·학습 규약", "  제2절 방어선 1: 시드 사전 등록 앙상블", "  제3절 방어선 2: 앙상블 성과와 점수 셔플 대조", "  제4절 방어선 3: 확정 뒤 1회 홀드아웃", "  제5절 가설 3: 방향은 CNN, 크기는 발생 모델",
       "제6장 논 의", "  제1절 왜 열지도인가", "  제2절 파라미터 민감도: 왜 14일과 4시간봉인가", "  제3절 실무 함의", "  제4절 한계",
       "제7장 결 론", "참고문헌", "부록 A 실전 페이퍼 장부", "부록 B 재현 절차와 결과 원자료", "부록 C 펀딩비 반영 상세", "부록 D 부수 실험: 급등·급락 발생 예측", "부록 E 보유 기간과 국면별 성과"]
for t in TOC: para(t, 10.5, t.startswith("제") or t.startswith("참") or t.startswith("부"), "left", indent=False, after=2)

# ================= 제1장 =================
chapter("제1장 서 론")
section("제1절 연구 배경과 선행연구")
para("시계열을 그림으로 바꿔 신경망에 맡기는 발상은 이미지 인식에서 검증된 합성곱 신경망을 그대로 쓰려는 데서 나왔다. Wang·Oates(2015)는 시계열을 그라미안 각도장과 마르코프 전이장 이미지로 바꿔 CNN 으로 분류했고 Sezer·Ozbayoglu(2018)는 기술지표 15개와 15일을 15×15 이미지로 놓아 다우 30 종목과 ETF 의 매수·매도·보유를 학습시켰다. 금융 횡단면에 이 틀을 본격적으로 들여온 것이 Jiang·Kelly·Xiu(2023, 이하 JKX)다. CRSP 일봉 1993~2019년을 5·20·60일 창의 흑백 OHLC 차트(이동평균·거래량 포함)로 그려 5×3 합성곱과 2×1 풀링으로 이루어진 소형 CNN 에 학습시키자 이미지 신호가 모멘텀·반전 지표보다 나았고 8년 학습 뒤 19년 시험에서도 유지됐다. 다섯 모델 앙상블을 썼고 2×2 풀링이나 ReLU, 층 축소는 성능을 떨어뜨렸다고 적었다.")
para("같은 틀은 여러 시장으로 번졌다. 중국 A주 연구(SSRN 5136032, 2025)는 2채널 그레이스케일로 확장해 거래량·과거수익·이동평균 채널이 성능을 올린다고 했다. 암호화폐 연구(arXiv 2605.00875, 2026)는 야후 일봉 BTC·ETH 에 사전학습 비전 모델을 파인튜닝해 4~16% 개선을 보였지만 종목 수가 둘이고 거래량을 넣으면 오히려 나빠졌으며 단순 CNN 이 ResNet·ViT 보다 나았다. 세 연구 모두 같은 정보를 표로 받은 대조군과 거래비용이 없다.")
para("이미지와 별개로 구조화 피처로 횡단면을 예측하는 기계학습에는 이미 두꺼운 계보가 있다. Gu·Kelly·Xiu(2020)는 주식 특성 94개로 트리와 신경망이 선형 모델을 이기며 그 이득이 비선형성과 상호작용에서 온다고 보였고 Krauss·Do·Huck(2017)은 S&P 500 통계적 차익거래에서 심층신경망·그래디언트 부스팅·랜덤 포레스트의 앙상블이 비용 전 하루 0.45% 를 내되 시간이 갈수록 줄어드는 것을 기록했다. 암호화폐 횡단면은 Liu·Tsyvinski·Wu(2022)가 시장·규모·모멘텀 3요인으로 설명했고 모멘텀은 Jegadeesh·Titman(1993) 이래 가장 오래된 횡단면 신호다. 이 연구가 대조군으로 LightGBM(Ke 외 2017)을 두고 피처에 Amihud(2002) 비유동성을 넣은 것은 이 계보를 따른 것이다.")
para("암호화폐는 이미지 접근이 가장 그럴듯해 보이는 시장이다. 종목이 수백 개이고 24시간 거래되며 개인 투자자 비중이 높아 차트 형태가 매매 행동에 직접 영향을 준다는 통념이 강하다. 그런데 기존 연구는 BTC·ETH 두 종목의 일봉 추세 분류에서 멈췄다. 수백 종목을 같은 시각에 줄 세우는 횡단면 예측, 곧 어느 종목이 다른 종목보다 오를지를 이미지로 맞히는 문제는 비어 있었다. 본 연구는 이 공백을 다루는 과정에서 예상과 다른 결과를 얻었다. 주식에서 유효했던 캔들 차트 이미지는 암호화폐 횡단면에서 유의한 신호를 내지 못하였다. 실패 원인을 분석하고 이미지 표현을 재설계한 결과 동일한 CNN 이 같은 정보를 수치로 받은 구조화 대조군을 넘어섰다. 본 논문은 이 과정을 실험 순서에 따라 기술한다.")
para("결과를 믿을 수 있게 만드는 규약도 선행연구에서 가져왔다. Harvey·Liu·Zhu(2016)는 수백 개 요인이 발표된 뒤라면 t 값 3 은 넘어야 새 요인이라 부를 수 있다고 했고 Hou·Xue·Zhang(2020)은 보고된 이상현상 대부분이 재현되지 않는다고 했다. López de Prado(2018)는 시계열 교차검증의 정제와 엠바고를 제안했다. 이 연구가 시드 열 개를 사전 등록하고 창 길이와 지평만큼 엠바고를 두며 확정 뒤 홀드아웃을 한 번만 쓰고 점수 셔플 대조를 함께 보고하는 것은 그 때문이다.")
section("제2절 가설 0: 캔들 차트 이미지의 스물다섯 설정")
para("연구는 이 통념을 가설 0 으로 설정하고 출발하였다.")
para("가설 0. 주식에서 성공한 캔들·거래량 차트 이미지는 암호화폐 횡단면 방향 예측에서도 작동한다.", bold=True, indent=False, before=4, after=4)
para("JKX 를 따라 창 안 OHLC 캔들과 거래량 막대를 96px 높이, 봉당 3px 폭의 그레이스케일로 그렸다({F:candle}). 여기서 출발해 격자와 창·지평(일봉 20봉·60봉, 4시간봉 20·60·120봉, 1시간봉 60봉), 채널(흑백, 거래 밀도·시각 평면을 더한 3채널, BTC 매크로 평면, 30일 거래량·요일까지 더한 5채널), 거래량 유무, 추세 제거, 해상도(높이 64px, 봉 폭 5px), 종목 수(100·200·400), 학습률(1e-3·1e-4), 유니버스(현재 시점·판단 시점 유동성)를 바꾼 25개 설정을 각각 롤링 4폴드로 돌렸다({T:settings}). 판정 잣대는 시험 구간의 상대방향 점수 차 spread_z(상위 십분위와 하위 십분위의 선행 상대수익 차를 표준화한 값)와 상·하위 십분위 적중률이고 채택 기준은 결과를 보기 전에 spread_z 0.2 이상을 4폴드 중 3폴드로 정해 두었다. 같은 폴드·같은 유니버스의 LightGBM 이 0.12~0.36 이므로 0.2 는 대조군의 낮은 폴드 수준이다.")
figure("fig2_candle.png", "가설 0 의 입력: JKX 형 캔들·거래량 그레이스케일 이미지 (예시 두 종목, 4h·60봉)@@candle", 13.5)
cs = pd.read_csv(f"{OUTDIR}/candle_settings_table.csv")
table("가설 0 의 캔들 차트 이미지 설정 25개 (JKX 형 OHLC 막대 + 거래량, 96px 높이, 봉당 3px, 소형 CNN, 상대수익 부호 라벨, 롤링 4폴드)@@settings", ["번호", "격자·창·지평", "채널", "변형"], [[str(i + 1), r.grid, r.channels, r.variant] for i, r in cs.iterrows()], widths=[1.0, 2.6, 5.4, 6.2], size=8, note="격자·창·지평 = 봉 간격·입력 창 봉수·라벨 지평 봉수(1d·20·20 = 일봉 20봉 창, 20일 지평). 채널의 밀도 = 봉별 거래량 밝기 평면, 시각·요일 = 판단 시각 평면, BTC 매크로 = BTC 24시간·7일 수익과 시장폭 평면. 변형이 없는 행은 기본(거래량 포함, 높이 96px, 봉 폭 3px, 종목 200, 학습률 1e-3, 현재 시점 유동성 유니버스).")
para("실험 결과 가설 0 은 사전 등록 기준에 미달하여 기각되었다. 25개 설정 100개 폴드 값이 −0.22~+0.16 에 머물렀고 기준을 넘긴 설정은 없었으며 상위 십분위 적중률은 설정 평균 41~48% 로 모두 절반 아래였다. 이에 연구 질문을 CNN 이 유효한가에서 CNN 에 제시한 이미지가 적절한가로 전환하였다. 실패를 해부하고(제3장) 그림을 다시 설계했으며(제4장) 새 그림으로 소형 CNN 이 같은 정보의 구조화 대조군을 넘는 것을 시드·셔플·홀드아웃으로 검증했다(제5장).")
section("제3절 이 연구의 차이, 기여와 구성")
para("차트 이미지 선행연구 셋과 이 연구의 차이는 {T:prior} 과 같다. 데이터는 상장폐지를 포함한 740종목 1분봉이고 라벨은 절대수익이 아니라 같은 시각 유니버스 평균을 뺀 상대수익 부호이며 대조군으로 같은 정보량의 LightGBM 을 두었다. 검증은 학습 1회·시험 1회가 아니라 롤링 4폴드와 확정 뒤 홀드아웃이고 거래비용을 반영한 코호트 롱숏으로 판정한다. 세 연구 모두 표 피처 대조군과 거래비용이 없었다.")
table("선행 연구와의 차이@@prior", ["항목", "JKX (JF 2023)", "중국 A주 (SSRN 2025)", "암호화폐 (arXiv 2026)", "이 연구"], [
 ["데이터", "CRSP 일봉 1993~2019", "A주 일봉", "야후 일봉 BTC·ETH", "무기한선물 1분봉 3.4년, 740종목, 상폐 포함"],
 ["라벨", "5·20·60일 수익 부호", "고저 십분위", "7일 ±2% 이진", "14일 상대수익 부호(유니버스 평균 차감)"],
 ["이미지", "흑백 OHLC·이동평균·거래량", "2채널 그레이스케일", "캔들 RGB", "캔들(기각) → 수치 열지도 + 횡단면 순위 + 피처 행"],
 ["모델", "소형 CNN, 5모델 앙상블", "CNN", "단순 CNN > ResNet > ViT", "소형 CNN 75만 파라미터, 시드 10개 앙상블"],
 ["대조군", "없음", "없음", "없음", "같은 정보량 LightGBM, 같은 정보의 MLP"],
 ["검증", "8년 학습 후 19년 시험", "미상", "70/15/15 분할", "롤링 4폴드 + 확정 뒤 홀드아웃 + 시드 사전 등록 + 셔플"],
 ["비용", "없음", "없음", "없음", "왕복 0.2%, 실측 펀딩비, 코호트 롱숏"]], widths=[1.6, 3.2, 3.0, 3.2, 5.0], size=8)
para("기여는 셋이다. 첫째, 캔들 차트 가설을 25개 설정 100개 폴드에서 체계적으로 기각하고 원인 둘을 규명한다. 모델은 형태 대신 모멘텀과 왜도를 외웠고 창 안 정규화가 횡단면 순위·장기 거래량·시각 맥락을 지웠다. 둘째, CNN 이 읽을 수 있는 차트 이미지(heatf)를 제안하고 단계별 제거 실험과 같은 정보를 받은 MLP 대조군으로 어느 행이, 그리고 2D 합성곱이 알파에 얼마나 기여하는지 보인다. 셋째, 시드 사전 등록·롤링 폴드·확정 뒤 1회 홀드아웃·점수 셔플 대조로 이루어진 평가 규약을 제시하고 시드 열 개 전부를 보고한다. 넷째, 방향 예측(CNN)과 크기·변동성 예측(발생 모델)을 결합한 2단계 비중 배분이 단독 방향 모델보다 유의한 추가 알파를 냄을 보인다.")
para("제2장은 기각과 채택에 공통으로 쓴 데이터·라벨·평가 규약과 두 모델을 고른 이유다. 제3장이 실패의 해부, 제4장이 표현의 재설계, 제5장이 채택과 검증, 그리고 방향과 크기를 다른 모델에 맡기는 2단계 결합(가설 3)이며 제6장에서 왜 열지도인가와 파라미터 민감도, 실무 함의·한계를 논의하고 제7장에서 맺는다. 부록은 실전 장부, 재현 절차, 펀딩비 상세, 급등·급락 발생 예측, 보유 기간·국면별 성과다.")

# ================= 제2장 =================
chapter("제2장 공통 잣대: 데이터, 라벨, 평가 규약")
para("가설 0 의 기각과 새 그림의 채택은 같은 데이터·같은 라벨·같은 잣대로 판정했다. 그 잣대를 먼저 적는다.")
section("제1절 데이터와 생존편향 실측")
para(f"여섯 거래소(binance·bybit·hyperliquid·grvt·paradex·backpack)의 USDT 무기한선물 1분봉을 2023-01 부터 수집한 CryptoBars 저장소를 쓴다. 저장소는 기간 중 한 번이라도 거래된 {N_ALL:,}종목 {ROWS_ALL/1e8:.2f}억 행이고 상장폐지 종목 {DEL_ALL}개를 포함한다. 거래소가 공식 봉을 주지 않는 상장폐지 종목은 체결 자료를 1분봉으로 다시 집계했다. 토큰화 주식·원자재와 기간 요건에 못 미치는 종목을 뺀 {N_USE}종목({ROWS_USE/1e8:.2f}억 행, 상장폐지 {DEL_USE}종목 포함)이 연구 대상이며 데이터 창은 2023-01 부터 2026-05 까지다. 1분봉을 4시간봉으로 묶고 판단 시각은 4시간봉 마감, 입력 창은 직전 60봉(10일)이다.")
table("데이터 요약@@data", ["구분", "종목", "행(1분봉)", "상장폐지", "출처"], [
 ["저장소 전체", f"{N_ALL:,}", f"{ROWS_ALL:,}", f"{DEL_ALL}", f"binance {SRC_ALL.get('binance',0)} · bybit 공식 봉 {SRC_ALL.get('bybit',0)} · bybit 체결 재집계 {SRC_ALL.get('bybit_trades',0)}"],
 [("연구 대상",), (f"{N_USE}",), (f"{ROWS_USE:,}",), (f"{DEL_USE}",), f"binance {SRC_USE.get('binance',0)} · bybit 공식 봉 {SRC_USE.get('bybit',0)} · bybit 체결 재집계 {SRC_USE.get('bybit_trades',0)}"],
 ["폴드 유니버스", "200", "—", "폴드별 상이", "판단 시점 직전 12개월 달러 거래대금 상위 200"]], widths=[2.4, 1.6, 2.6, 1.8, 7.4], note="출처: data/coverage.csv. 기간 2023-01-01~, 시간대 UTC. 체결 재집계 봉은 거래소 공식 봉과 끝자리가 다를 수 있다.")
para("생존편향은 이 데이터에서 직접 쟀다. 유니버스를 현재 시점 유동성 상위 100 으로 고르면 판단 시점 직전 12개월 유동성으로 고를 때보다 상대방향 점수 차가 크게 부풀었다({T:univ}). 살아남아 거래대금이 커진 종목만 남기 때문이다. 그래서 모든 폴드의 유니버스는 판단 시점 직전 12개월 달러 거래대금 상위 200 종목으로 폴드마다 따로 만든다. 폴드가 미래를 보지 않으므로 상장폐지 종목이 자연스럽게 들어온다.")
table("유니버스 선정 기준에 따른 LightGBM 상대방향 점수 차(spread_z, 폴드 0~3)@@univ", ["격자·창·지평", "판단 시점 12개월 상위 100", "전체 종목", "현재 시점 상위 100 (미래 정보)"], [
 ["1d·20·20", "0.24 · 0.41 · 0.41 · 0.09", "0.16 · 0.24 · 0.32 · 0.14", "0.36~0.92"],
 ["4h·60·42", "0.18 · 0.29 · 0.15 · 0.22", "0.22 · 0.18 · 0.10 · 0.28", "0.36~0.92"]], widths=[2.6, 4.4, 4.0, 4.6], note="현재 상장 종목만으로 과거를 되돌아보면 신호가 2~4배 부풀어 보인다. 이 연구의 모든 표는 판단 시점 기준 유니버스다.")
section("제2절 라벨과 지평")
para("판단 시각 t 에 종목 i 의 14일(84봉) 선행 로그수익에서 같은 시각 유니버스 평균을 뺀 값을 상대수익으로 정의한다. 라벨은 그 부호(상승·하락)이고 모델은 상승·하락 확률을 내며 점수는 두 확률의 차다. 시장 평균을 빼면 BTC 방향과 시장 베타가 사라지고 횡단면 순위만 남는다. 지평 14일은 7·21일보다 폴드 간 일관성이 높아 택했다. 7일 지평은 폴드 0·1 에서 Sharpe 4.02·0.25 로 흔들렸고 21일은 0.51 이었다. 1~5봉의 짧은 지평에서는 어떤 표현도 방향을 잡지 못했다.")
section("제3절 폴드와 홀드아웃")
para("학습(IS)·검증(OS)·시험(ROS) 3개월 롤링 폴드 네 개를 쓴다. 선택은 검증에서만, 판정은 시험에서만 한다. 학습은 2023-01 부터 검증 직전까지이고 창 길이와 지평만큼 엠바고를 둔다. 홀드아웃은 그림·모델·지평을 네 폴드로 확정한 뒤 한 번만 썼다.")
table("폴드 구성@@folds", ["폴드", "학습(IS)", "검증(OS)", "시험(ROS)", "시험 구간 성격"], [
 ["A", "2023-01~2024-02", "2024-03~05", "2024-06~08", "알트 약세, BTC −13.9%"], ["B", "2023-01~2024-08", "2024-09~11", "2024-12~2025-02", "알트 약세, BTC −13.4%"],
 ["C", "2023-01~2025-05", "2025-06~08", "2025-09~11", "알트 약세, BTC −17.5%"], ["D", "2023-01~2025-08", "2025-09~11", "2025-12~2026-02", "알트 약세, BTC −25.4%"],
 [("홀드아웃",), "2023-01~2025-11", "2025-12~2026-02", ("2026-03~05",), "설정 확정 뒤 첫 사용"]], widths=[1.6, 3.2, 2.6, 3.0, 5.2], note="시험 구간 성격의 BTC 수익은 각 구간 시작·끝 종가 기준. 유니버스 평균 14일 로그수익은 네 폴드에서 −6.7~−14.3% 였다.")
section("제4절 코호트 롱숏과 판정 지표")
para("판단 시각마다 점수 상위 10% 를 매수, 하위 10% 를 매도해 14일 보유하는 달러중립 코호트를 연다. 겹치는 코호트에 자본을 균등 배분하고 왕복 비용 0.2% 를 보유 기간에 선형 차감하며 수익은 단순수익으로 잰다.")
para("로그수익 기반 점수 차(spread)는 공매도 다리의 왜도 때문에 손익을 과대평가한다. 하위 십분위에는 가끔 2~3배 뛰는 종목이 있어 로그로는 손실이 −69% 로 잘리지만 실제 공매도 손실은 −100% 를 넘는다. 같은 코호트를 로그로 재면 교사 모델이 14일당 +213bp 였지만 단순수익으로는 +75bp 였다. 그래서 spread_z 는 탐색 단계에서 신호 유무를 가르는 잣대로만 쓰고 성과 판정은 단순수익 손익으로 한다.")
para("지표는 일별 수익의 연율 Sharpe(√365), 평균 일수익이 0 이하라는 귀무가설에 대한 Newey-West(지연 14일) 단측 p, 최대낙폭(MDD)이다. 대조로 판단 시각마다 점수를 종목 간 무작위로 섞은 셔플을 폴드마다 두 번 돌려 같은 규칙으로 평가한다.")
para("무기한선물의 펀딩비는 기본 판정에서 빼고 따로 반영해 보고한다. 바이낸스 8시간 정산 이력(2024-05 이후)을 보유 중 누적해 롱 다리는 내고 숏 다리는 받는 방향으로 손익에 넣는다. 펀딩 자료가 있는 종목은 폴드별 유니버스의 62~118개이고 자료가 없는 종목(bybit 전용 상장 등)은 같은 다리의 평균 펀딩을 적용한다.")
section("제5절 대조군 LightGBM")
para("같은 유니버스·라벨·평가로 LightGBM 을 학습한다. 입력은 27개 수치 피처(수익·변동성·범위·이동평균 관계·RSI·거래량 비율·Amihud·거래 집중도·급등 횟수·시각·요일 등)와 10개 횡단면 백분위 순위이며 창 길이·잎 수가 다른 세 모델(4h·60·84, 4h·60·42, 4h·120·42)의 표준화 평균을 쓴다(v2). 이 대조군은 24개월 연속 walk-forward 에서 단독 Sharpe 1.92(p 0.0046), 7일 지평 모델(v1)과 자본을 반씩 섞으면 2.39(p 0.0016)로 그 자체로 유효한 전략이다({T:wf}, {F:wf}). 이미지 CNN 이 이 기준을 넘는지가 가설 2 의 판정이다. 대조군은 CNN 의 경쟁자가 아니라 같은 정보를 수치로 받았을 때의 기준선이다. 그림이 정보를 살리지 못하면 CNN 은 이 선 아래에 머물고(캔들), 살리면 넘는다(열지도). 이 논문이 재는 것은 모델의 우열이 아니라 그림의 우열이다.")
figure("fig11_wf.png", "LightGBM 상대방향 전략의 8폴드 walk-forward Sharpe@@wf", 14.0)
table("LightGBM 상대방향 전략의 분기 연속 walk-forward (시험 2024-03~2026-02, 8폴드 24개월)@@wf", ["전략", "폴드별 Sharpe (2024-03 → 2025-12 분기)", "합산 Sharpe (NW p)", "MDD", "양수 폴드"], [
 ["v1 (7일 지평)", "−0.98 · 3.91 · 2.69 · 0.86 · 2.85 · 3.37 · −0.43 · 2.90", "1.58 (0.014)", "−8.1%", "6/8"],
 ["v2 (14일 지평)", "−1.48 · 2.98 · 4.45 · 5.05 · 0.15 · 5.65 · 2.77 · −1.60", "1.92 (0.0046)", "−10.5%", "6/8"],
 [("v1+v2 자본 50/50",), "−1.70 · 4.58 · 4.79 · 3.70 · 2.37 · 6.45 · 1.25 · 1.30", ("2.39 (0.0016)",), "−8.2%", "7/8"]], widths=[2.8, 7.0, 2.6, 1.4, 1.8], note="폴드별 유동성 상위 200, K=100 롱숏, 왕복 0.2%, 판단 시각별 횡단면 십분위 경계. 첫 폴드(2024-03~05)는 학습 11개월뿐인 유일한 공통 음수.")
section("제6절 왜 LightGBM 이고 왜 소형 CNN 인가")
para("대조군을 LightGBM 으로 둔 이유는 셋이다. 첫째, 표 형태 특성으로 횡단면을 예측할 때 그래디언트 부스팅 트리는 Gu·Kelly·Xiu(2020)와 Krauss·Do·Huck(2017)에서 신경망과 함께 가장 강한 축에 들었고 비선형성과 상호작용을 자동으로 잡는다. 둘째, 피처 스케일에 둔감하고 학습이 빨라 폴드마다 다시 학습하는 롤링 검증에 맞는다. 셋째, 피처 중요도가 나와 이미지가 무엇을 놓치는지 거꾸로 짚어 준다. 제3장 제3절의 원인 2 는 이 중요도에서 나왔다. 대조군이 약하면 이미지의 우위가 과장되므로 같은 피처로 만들 수 있는 가장 강한 모델을 두는 것이 이 연구의 원칙이고 제5절의 walk-forward 는 그 대조군이 그 자체로 유효한 전략임을 보인다.")
para("이미지 쪽을 75만 파라미터 소형 CNN 으로 둔 이유도 셋이다. 첫째, JKX 와 암호화폐 선행연구가 모두 작은 CNN 이 큰 백본을 이겼다고 보고했고 이 연구의 사전학습 ResNet18 직접 학습도 실패했다(제4장 제4절). 학습 창이 폴드당 50만 개 안팎이라 큰 모델은 외운다. 둘째, 2×1 풀링으로 폭(시간)을 보존하는 JKX 구조가 봉 순서 정보를 끝까지 남긴다. 셋째, CPU 추론이 가능해 실전 장부(부록 A)에서 4시간마다 200 종목을 그리고 점수를 내는 데 문제가 없다. 시퀀스 트랜스포머와 이미지·피처 융합 모델도 같은 잣대로 시험했으나 LightGBM 을 넘지 못했고 표현을 바꾼 소형 CNN 만이 넘었다.")

# ================= 제3장 =================
chapter("제3장 실패의 해부: 캔들 차트는 왜 실패했는가")
section("제1절 스물다섯 설정의 시험 결과")
para("시험 결과는 {T:candle} 과 {F:candleset} 에 제시한다. 25개 설정 100개 폴드 값이 −0.22~+0.16 사이에 있고 평균은 −0.01 이며 절반이 넘는 54개가 음수다. 사전 등록 기준(spread_z 0.2 이상을 4폴드 중 3폴드)에 든 설정은 하나도 없고 4폴드 전부 양수인 설정도, 4폴드 평균이 0.05 를 넘는 설정도 없다. 가장 좋은 설정(1d·20·20 흑백, 종목 100)의 평균이 +0.04 이고 가장 나쁜 설정(1d·60·20 추세 제거)은 −0.10 이다. 같은 폴드·같은 유니버스의 LightGBM 은 0.12~0.36 이다. 상위 십분위 적중률은 설정 평균 41~48% 로, 이미지가 가장 오를 것 같다고 고른 열 종목 가운데 다섯에서 여섯이 상대적으로 내렸다는 뜻이다. 거래량 제외, 창 축소, 채널 추가, 격자·종목 변경 어느 경우에도 부호는 회복되지 않았다. 폴드별로는 폴드 3(2024-06~08)만 설정 평균 +0.05 로 양수였고 나머지 세 폴드는 음수였다. 검증(OS) 구간에서는 0.15~0.25 로 보이던 값이 시험(ROS)으로 옮겨가지 않았다는 점도 같다. 검증 구간에서 선택한 설정이 시험 구간에서 무너지는 전형적인 선택 편향이다.")
crows = [[f"{r.grid}" + ("" if r.channels == "흑백" else " " + r.channels) + ("" if r.variant == "—" else ", " + r.variant), f"{r.f0:+.3f}", f"{r.f1:+.3f}", f"{r.f2:+.3f}", f"{r.f3:+.3f}", f"{r['mean']:+.3f}", f"{r.hit_top_mean:.2f}"] for _, r in cs.iterrows()]
crows += [[("LightGBM 1d·20·20 (대조군)",), "+0.117", "+0.184", "+0.255", "+0.143", ("+0.175",), "0.48~0.59"], [("LightGBM v2 4h·60·84 (대조군)",), "+0.26", "+0.35", "+0.36", "+0.34", ("+0.33",), "0.52~0.57"]]
table("캔들 이미지 CNN 25개 설정의 시험 구간 spread_z (폴드 0~3)와 상위 십분위 적중률@@candle", ["설정", "폴드 0", "폴드 1", "폴드 2", "폴드 3", "평균", "상위 적중"], crows, widths=[6.4, 1.5, 1.5, 1.5, 1.5, 1.5, 1.7], size=7.5, note="폴드 0 은 D(2025-12~02), 1 은 C, 2 는 B, 3 은 A 로 실험 순서를 따른다. 상위 적중 = 상위 십분위 종목이 상대적으로 오른 비율의 4폴드 평균(대조군은 범위). 사전 등록 기준은 spread_z 0.2 이상을 4폴드 중 3폴드.")
figure("fig3_candle_settings.png", "캔들 이미지 CNN 설정별 시험 구간 spread_z (검정 점)와 LightGBM 대조군 (빈 원)@@candleset", 15.0)
section("제2절 원인 1: 모멘텀과 왜도의 암기")
para("먼저 spread_z 가 0 근처에 머문 원인을 분석하였다. 1d·20·20 흑백 CNN 의 점수와 직전 20일 수익률의 스피어만 상관은 폴드별 +0.32·−0.44·−0.44·−0.72 로 절대값이 크고 부호가 폴드마다 뒤집힌다. 20일 모멘텀 자체의 시험 구간 spread_z 도 −0.19·+0.09·+0.17·+0.08 로 분기마다 부호가 바뀐다({F:momentum}). 차트 형태가 담는 것은 결국 과거 수익 경로이고 암호화폐에서 20일 모멘텀의 예측 부호는 체제 의존이다. 이미지 모델은 학습 구간의 부호를 학습하여 다음 분기에 오류를 낸다. 같은 폴드의 LightGBM 점수는 모멘텀과의 상관이 |ρ| ≤ 0.3 이고 스프레드를 거래 밀도·변동성 피처에서 얻는다.")
figure("fig4_momentum.png", "캔들 CNN 점수의 모멘텀 의존과 모멘텀 부호의 체제 의존 (1d·20·20, 폴드 0~3)@@momentum", 14.5)
para("십분위별 분해는 두 번째 원인을 드러낸다. CNN 은 상위 십분위도 하위 십분위도 평균 상대수익이 양수(+0.01~+0.17)인데 적중률은 상위 40~49%·하위 53~61% 다. 양 끝 모두 대개는 내리지만 가끔 크게 오르는 오른쪽 꼬리 종목을 고른 것이다. 이미지 점수는 방향이 아니라 왜도, 곧 복권형 수익 분포를 선별한 것이다. LightGBM 은 상위 +0.06~+0.27, 하위 −0.01~−0.09 로 부호가 갈린다({T:decile}).")
table("십분위 해부: 캔들 CNN 과 LightGBM 의 상·하위 십분위 (4폴드 범위)@@decile", ["모델", "상위 십분위 평균 상대수익(z)", "상위 적중률", "하위 십분위 평균 상대수익(z)", "하위 적중률", "읽기"], [
 ["캔들 CNN (1d·20·20 흑백)", "+0.01~+0.17", "40~49%", "+0.01~+0.17", "53~61%", "양 끝 모두 오른쪽 꼬리 종목"],
 ["LightGBM (1d·20·20)", "+0.06~+0.27", "48~59%", "−0.01~−0.09", "55~63%", "부호가 갈린다"]], widths=[3.6, 2.8, 1.8, 2.8, 1.8, 3.2], size=8)
section("제3절 원인 2: 창 안 정규화가 지운 맥락")
para("캔들 이미지는 창 안 최고·최저로 정규화된다. 이 정규화가 세 가지를 지운다. 이 종목이 같은 시각 유니버스 안에서 어디에 있는지(횡단면 순위), 이 거래량이 30일 평소보다 많은지(장기 맥락), 이 봉이 어느 시각·요일인지(계절성)다. 셋 다 제2장 대조군의 상위 중요도 피처였다. 특히 시각·요일은 급등·급락 발생 예측에서 가장 큰 신호였는데 이미지 평면으로 더해도 +0.02 안팎밖에 못 썼다. 이미지에 없는 정보를 CNN 이 학습할 수는 없다. 같은 데이터로 한 급등·급락 발생 예측(부록 D)에서도 맥락 평면을 더한 5채널 이미지가 흑백을 4폴드 중 3폴드에서 이겼고 이미지 단독은 피처 모델을 넘지 못했다. 맥락 부재의 효과는 과제를 바꾸어도 동일하게 나타난다.")
section("제4절 이미지는 구조화 모델에 보태지 않았다")
para("마지막으로 이미지 점수가 보조 신호로서 기여하는지를 검정하였다. 검증과 시험을 모두 덮는 CNN 점수를 LightGBM 점수와 z-혼합하고 가중치는 검증에서 고른 뒤 시험에서 종목×월 블록 부트스트랩으로 순기여를 쟀다({T:blend}). 1d·20·20 흑백은 폴드 0 에서 Δspread_z −0.005(95% 구간 −0.065~+0.036), 폴드 1 에서 +0.001(−0.080~+0.058), 5채널은 −0.010(−0.045~+0.045)이고 검증에서 선택된 최적 가중치는 0 이었다. 4h·60·42 흑백은 −0.025(−0.056~+0.001)이다. 이미지는 방향 예측에 기여하지 않았다. 이 실행의 유니버스는 현재 유동성 기준이라 생존편향이 이미지에 유리한 조건이었는데도 그렇다.")
table("캔들 CNN 점수를 LightGBM 에 z-혼합했을 때의 순기여 (Δspread_z, 시험 구간, 95% 부트스트랩 구간)@@blend", ["설정", "폴드 0", "폴드 1", "검증 최적 가중치"], [
 ["1d·20·20 흑백", "−0.005 (−0.065~+0.036)", "+0.001 (−0.080~+0.058)", "0"],
 ["1d·20·20 5채널", "−0.010 (−0.045~+0.045)", "—", "0"],
 ["4h·60·42 흑백", "0 (혼합이 단독을 넘지 못함)", "−0.025 (−0.056~+0.001)", "0"]], widths=[3.4, 4.6, 4.6, 2.6], size=8.5)
para("이상으로 가설 0 은 기각되었다. 남는 질문은 모멘텀·왜도 대신 맥락을 학습하게 하려면 무엇을 그려야 하는가이다.")

# ================= 제4장 =================
chapter("제4장 표현의 재설계: CNN 이 읽을 수 있는 열지도(heatf)")
section("제1절 원인 둘을 뒤집는 가설 1")
para("제3장이 찾은 원인은 둘이다. 원인 1 은 모델이 형태 대신 창 안 모멘텀과 수익률 왜도를 외운다는 것이다. 캔들 그림에서 CNN 이 읽을 수 있는 가장 강한 규칙성은 최근 수익 경로이고 암호화폐에서 그 부호는 분기마다 뒤집히므로 학습 구간의 부호를 외운 모델은 다음 분기에 틀린다. 십분위 양 끝이 모두 오른쪽 꼬리 종목으로 채워진 것도 같은 원인이다. 원인 2 는 창 안 최고·최저로 정규화하는 캔들 그림이 맥락을 지운다는 것이다. 이 종목이 같은 시각 유니버스 안에서 어디에 있는지, 이 거래량이 30일 평소보다 많은지, 이 봉이 어느 시각·요일인지가 그림에 없고 그림에 없는 것을 CNN 이 배울 수는 없다. 두 원인을 뒤집으면 처방이 도출된다. 정규화 기준을 창 밖에 두어 스케일과 맥락을 보존하고 횡단면 순위와 장기 거래량, 시각을 행으로 그림 안에 넣는다. 모멘텀은 지우지 않되 그것만 보이는 그림은 피한다.")
para("가설 1. 창 안 수치를 행으로 쌓되 30일 평균·유니버스 백분위 순위·학습 구간 통계로 정규화한 열지도 이미지를 주면 소형 CNN 이 횡단면 알파를 복원한다.", bold=True, indent=False, before=4, after=4)
section("제2절 heatf 의 구성")
para("48행을 세로로 두 번 반복해 96px, 60봉을 봉당 3px 로 180px 폭이다({F:heatf}). 행은 다섯 묶음이다.")
bullet("가격 4행. 창 마지막 종가 대비 O·H·L·C 로그비를 ±10% 로 잘라 0~1 로 놓는다. 캔들처럼 창 안 최고·최저가 아니라 마지막 종가가 기준이므로 스케일이 보존된다.")
bullet("거래량 2행. 창 평균 대비 거래량(0~3배)과 30일 평균 대비 거래량(±3σ). 두 번째 행이 평소보다 많은가를 담는다.")
bullet("시각·요일 2행. 봉의 UTC 시각과 요일.")
bullet("횡단면 순위 3행. 봉마다 24시간 수익·24시간 변동성·30일 대비 거래량 세 값의 같은 시각 유니버스 안 백분위 순위를 매긴다. 이 세 행이 지금 이 종목이 다른 종목보다 어디에 있는가를 그림 안에 넣는다.")
bullet("요약 피처 37행. 창 끝에서 계산한 27개 수치 피처와 10개 횡단면 순위를 학습 구간 통계로 표준화한 뒤 시그모이드로 0~1 에 놓고 창 폭 방향으로 반복한다. 표준화 통계는 학습 집합에서만 구하고 검증·시험·실전에 그대로 쓴다.")
para("가격·거래량·시각 8행만 그린 판(heat), 횡단면 순위까지 11행인 판(heatx), 피처 행까지 넣은 판(heatf)을 따로 두어 어느 행이 기여하는지 뗀다. 사후 정보는 없다. 피처는 창 안 과거이고 순위는 같은 시각 횡단면이며 표준화 통계는 학습 집합이다.")
figure("fig1_heatf.png", "heatf 이미지 예시: 최종 모델 점수 상위·하위 종목 (2026-05-31 판단 시각, 96×180 그레이스케일)@@heatf", 15.0)
section("제3절 단계별 제거 실험: 어느 행이 알파를 되살리는가")
para("같은 소형 CNN 에 그림만 바꿔 넣었다({T:ladder}). 8행 열지도(heat)만으로 4h·60·42 의 시험 구간 spread_z 가 0.21·0.09·0.03·0.15 로 올라 캔들의 0.05·−0.03·0.00·0.08 을 벗어났고 폴드 0 은 LightGBM(0.20) 수준이었다. 순위 3행을 더해 14일 지평으로 늘리자(heatx 84봉) 폴드 0·1 에서 0.28·0.33 으로 LightGBM 급이 됐고 코호트 롱숏 Sharpe 는 폴드 0 1.19(같은 분기 LightGBM 1.22), 폴드 1 0.67 이었다. 다만 4폴드 합산은 0.96 으로 폴드 2 가 −0.14 였다. 피처 37행을 더한 heatf 는 시드 0 하나로 4폴드 Sharpe 2.78·2.69·3.57·3.23, 합산 2.92 를 냈다. 순위 세 행만 더해도 부호가 양수로 돌아서고 피처 행이 더해지면 대조군 2.18 을 넘는다({F:ladder}).")
table("이미지 구성 사다리: 같은 소형 CNN, 교사 없이 라벨 직접 학습@@ladder", ["단계", "이미지", "행 구성", "시험 spread_z (폴드 0·1·2·3)", "코호트 Sharpe"], [
 ["0", "캔들 + 거래량 (25개 변형)", "픽셀 차트", "0.05 · −0.03 · 0.00 · 0.08 (4h·60·42)", "판정 미도달"],
 ["1", "heat", "가격·거래량·시각·요일 8행", "0.21 · 0.09 · 0.03 · 0.15 (4h·60·42)", "—"],
 ["2", "heatx", "8행 + 횡단면 순위 3행", "0.15 · 0.08 · −0.02 · — (42봉) / 0.28 · 0.33 (84봉)", "폴드 0 1.19 · 폴드 1 0.67 · 합산 0.96"],
 [("3",), ("heatf",), "11행 + 창 끝 피처 37행", "0.13 · 0.30 · 0.21 · — (84봉)", ("2.78 · 2.69 · 3.57 · 3.23 · 합산 2.92",)],
 ["참고", "LightGBM v2 (같은 피처를 표로)", "—", "0.26 · 0.35 · 0.36 · 0.34", "1.22 · 1.54 · 2.68 · 4.48 · 합산 2.18"]], widths=[1.0, 3.4, 3.4, 5.0, 3.4], size=8, note="시드 0 기준. 시드 열 개 앙상블의 heatf 결과는 제5장. 폴드 0 은 D(2025-12~02), 1 은 C, 2 는 B, 3 은 A 로 실험 순서를 따른다.")
figure("fig5_ladder.png", "이미지 구성 사다리의 4폴드 합산 Sharpe@@ladder", 14.5)
section("제4절 사전학습 백본과 증류")
para("큰 백본이 답인지도 확인했다. ImageNet 사전학습 ResNet18 은 같은 heatf 그림을 교사 없이 직접 배우지 못했다(4h·60·42 폴드 0 spread_z −0.02, heatf 폴드별 −0.01·+0.14·−0.10). 대신 LightGBM 점수를 교사로 두고 MSE 로 회귀시키는 증류에서는 살아났다. heatf ResNet18 학생은 분할 A(시험 2025-09~2026-02) Sharpe 1.40(p 0.077), 분할 B(2025-06~11) 1.18(p 0.125), 합산 379일 1.29(p 0.034)였고 열지도·heatf 시드 네 모델 앙상블은 합산 1.33(p 0.030, MDD −9.8%)이었다({T:distill}). 교사(1.64)에는 못 미쳤다. 소형 CNN 이 큰 백본보다 낫고 교사 없이 직접 배우는 편이 증류보다 낫다는 결론은 JKX 와 arXiv 2605.00875 의 관찰과 같다. 가설 1 은 채택되었다.")
table("증류 학생(ResNet18, 교사 = LightGBM v2 표본 밖 점수)과 직접 학습의 비교 (코호트 롱숏 14일, Sharpe / NW p / MDD)@@distill", ["학생", "분할 A (2025-09~2026-02)", "분할 B (2025-06~11)", "합산 379일"], [
 ["ResNet18 열지도(heat) 시드 0", "1.26 / 0.08 / −6.7%", "1.23 / 0.09 / −9.5%", "—"],
 ["ResNet18 heatf 시드 0", "1.40 / 0.077 / −5.5%", "1.18 / 0.125 / −10.2%", "1.29 / 0.034 / −10.2%"],
 ["heat·heatf 시드 0·1 네 모델 앙상블", "1.32", "1.34", ("1.33 / 0.030 / −9.8%",)],
 ["교사 LightGBM v2", "1.14 / 0.11 / −5.7%", "2.05 / 0.007 / −5.7%", "1.64 / 0.005 / −5.7%"],
 ["점수 셔플", "−2.6", "−4.8", "—"],
 [("소형 CNN heatf 직접 학습 (시드 0, 4폴드)",), "—", "—", ("2.92 / 0.0001 / −4.6%",)]], widths=[5.2, 3.6, 3.4, 3.4], size=8, note="분할 A·B 는 증류 실험용 6개월 시험 구간. 직접 학습 행은 제2장 4폴드 408일 기준이라 구간이 다르다.")

section("제5절 이미지가 본질인가: 같은 정보를 받은 MLP 대조군")
_mlp = f"{OUTDIR}/mlp_table.csv"
if os.path.exists(_mlp):
    mt = pd.read_csv(_mlp); g = lambda name: mt[mt.model == name].iloc[0]
    m1e, m2e, c3, c10, gb = g("m1 시드 3개 앙상블"), g("m2 시드 3개 앙상블"), g("heatf 소형 CNN 시드 3개 앙상블"), g("heatf 소형 CNN 시드 10개 앙상블"), g("LightGBM v2 (같은 피처 37개)")
    para("heatf 의 알파가 요약 피처 37행을 CNN 이 읽은 결과일 뿐이라면 같은 37개 값을 표로 받은 신경망도 같은 성과를 내야 한다. 그래서 두 대조군을 같은 폴드·같은 시드(0·1·2)·같은 코호트 규칙으로 돌렸다. m1 은 창 끝 피처 37개만 받는 MLP(256·64 은닉, LayerNorm, 드롭아웃 0.3, 2.7만 파라미터)이고 m2 는 heatf 의 수치 행렬 48×60 전체(픽셀을 원래 값으로 되돌린 2,880개)를 펴서 받는 MLP(75만 파라미터, 소형 CNN 과 같은 크기)다. m2 는 CNN 과 정보가 완전히 같고 2D 합성곱만 없다({T:mlp}, {F:mlp}).")
    verdict = ("시드 3개 앙상블의 4폴드 합산 Sharpe 는 m1 %.2f, m2 %.2f 로 같은 시드 수의 heatf CNN %.2f 보다 낮고 LightGBM %.2f 과 견주어도 우위가 없다. 같은 정보를 받아도 2D 합성곱이 봉과 행의 국소 상호작용을 잡는 편이 평평한 완전연결보다 낫다는 뜻이며 이미지 처리가 기여한다는 제6장 제1절의 해석을 뒷받침한다." % (m1e.pooled, m2e.pooled, c3.pooled, gb.pooled)
               if c3.pooled > max(m1e.pooled, m2e.pooled) + 0.3 else
               "시드 3개 앙상블의 4폴드 합산 Sharpe 는 m1 %.2f, m2 %.2f 로 같은 시드 수의 heatf CNN %.2f 과 큰 차이가 없다. 이 결과는 알파의 본질이 2D 합성곱이 아니라 무엇을 넣느냐(창 밖 정규화·횡단면 순위·요약 피처)에 있음을 뜻한다. 이 논문의 주장은 그에 맞춰 \"CNN 을 위한 표현 설계\" 로 두고 2D 구조의 기여는 제한적이라고 적는다." % (m1e.pooled, m2e.pooled, c3.pooled))
    para(verdict)
    mrows = []
    for _, r in mt.iterrows():
        b = ("앙상블" in r.model)
        cells = [r.model] + [("—" if pd.isna(r[k]) else f"{r[k]:.2f}") for k in "ABCD"] + [("—" if pd.isna(r.pooled) else f"{r.pooled:.2f}"), ("—" if pd.isna(r.p) else ("< 0.0001" if r.p < 0.0001 else f"{r.p:.4f}")), ("—" if pd.isna(r.mdd) else f"{r.mdd:.1f}%")]
        mrows.append([(c,) if b else c for c in cells])
    table("같은 정보를 받은 MLP 대조군과 heatf 소형 CNN 의 코호트 롱숏 Sharpe (시드 0·1·2)@@mlp", ["모델", "A", "B", "C", "D", "합산 408일", "NW p", "MDD"], mrows, widths=[6.6, 1.2, 1.2, 1.2, 1.2, 1.6, 1.7, 1.3], size=8, note="m1 = 창 끝 피처 37개만 받는 MLP, m2 = heatf 수치 행렬 48×60 전체를 펴서 받는 MLP. 같은 폴드·유니버스·라벨·코호트 규칙.")
    if os.path.exists(f"{FIG}/fig12_mlp.png"): figure("fig12_mlp.png", "같은 정보를 받은 MLP 대조군과 heatf CNN 의 폴드별 Sharpe (시드 3개 앙상블)@@mlp", 14.5)

# ================= 제5장 =================
chapter("제5장 채택과 검증: 교사 없는 소형 CNN 과 3중 검증")
section("제1절 가설 2 와 모델·학습 규약")
para("가설 2. heatf 로 그린 그림을 받은 75만 파라미터 소형 CNN 은 교사 없이 라벨을 직접 학습해 같은 정보를 수치로 받은 구조화 대조군(LightGBM)을 넘는 알파를 낸다. 우위의 원천은 CNN 이 아니라 그림이다.", bold=True, indent=False, before=4, after=4)
para("모델은 JKX 형 블록 셋이다. 5×3 합성곱, 배치 정규화, LeakyReLU(0.01), 2×1 최대 풀링이 한 블록이고 채널은 64·128·256 이다. 6×30 적응 풀링 뒤 완전연결층이 상승·하락·중립 세 확률을 낸다. 파라미터는 75.5만 개다. AdamW(가중치 감쇠 1e-4), 배치 256, 8 에포크, 검증 구간 평균정밀도가 가장 높은 에포크의 가중치를 채택한다. 학습 창은 종목당 균등 추출로 500만 개 상한이고 교사 점수는 쓰지 않는다. 학습은 A40 GPU 한 장에서 폴드당 15~25분이다.")
section("제2절 방어선 1: 시드 사전 등록 앙상블")
para("시드 편차가 크다는 것을 세 시드에서 먼저 확인했다. 폴드 D 에서 시드 0·1·2 의 Sharpe 가 2.78·0.22·1.12 였다. 그래서 시드 0~9 열 개를 결과를 보기 전에 정해 두고 전부 학습해 전부 보고한다. 배포·판정 단위는 시드별 점수를 판단 시각마다 표준화해 평균한 앙상블이며 최고 시드를 선별한 표는 제시하지 않는다.")
rows = []
for _, r in seed.iterrows():
    rows.append([str(int(r.seed)), fmt(r["2024-06~08_sharpe"]), fmt(r["2024-12~2025-02_sharpe"]), fmt(r["2025-09~11_sharpe"]), fmt(r["2025-12~2026-02_sharpe"]), fmt(r.pooled_sharpe), fmt(r.pooled_p, 4), f"{r.pooled_mdd:.1f}%", fmt(r["holdout_2026-03~05_sharpe"]), fmt(r["holdout_2026-03~05_p"], 3)])
num = seed[["2024-06~08_sharpe", "2024-12~2025-02_sharpe", "2025-09~11_sharpe", "2025-12~2026-02_sharpe", "pooled_sharpe", "holdout_2026-03~05_sharpe"]]
rows.append([("평균",)] + [(f"{v:.2f}",) for v in num.mean()[:5]] + ["", ""] + [(f"{num.mean().iloc[5]:.2f}",), ""])
rows.append([("표준편차",)] + [(f"{v:.2f}",) for v in num.std()[:5]] + ["", ""] + [(f"{num.std().iloc[5]:.2f}",), ""])
rows.append([("최소",)] + [(f"{v:.2f}",) for v in num.min()[:5]] + ["", ""] + [(f"{num.min().iloc[5]:.2f}",), ""])
table("시드별 결과 (열 개 전부, 코호트 롱숏 Sharpe)@@seeds", ["시드", "A", "B", "C", "D", "합산 408일", "합산 NW p", "합산 MDD", "홀드아웃 03~05", "홀드아웃 p"], rows, widths=[1.3, 1.3, 1.3, 1.3, 1.3, 1.9, 1.8, 1.7, 2.2, 1.7], size=8, note="A=2024-06~08, B=2024-12~2025-02, C=2025-09~11, D=2025-12~2026-02. 합산은 네 폴드 일별 수익을 이어 붙인 408일.")
para("합산 Sharpe 는 시드 열 개 모두 1.5 를 넘고(2.29~3.64) p 는 모두 0.003 이하다. 가장 약한 폴드 D 에서도 열 개 중 아홉이 1 을 넘는다. 시드 편차는 실제로 크다. 폴드 D 의 시드 1 은 0.22 이고 폴드 B 의 시드 6 은 7.05 다({F:seeds}). 시드 열 개는 같은 데이터를 쓰므로 독립 표본이 아니다. 이 표가 증명하는 것은 결과가 초기화 운에 달려 있지 않다는 점이고 유의성 자체는 다음 두 방어선이 지탱한다.")
figure("fig6_seeds.png", "시드 열 개의 Sharpe 분포와 시드 10개 앙상블 (폴드·합산·홀드아웃)@@seeds", 14.5)
section("제3절 방어선 2: 앙상블 성과와 점수 셔플 대조")
para("{T:ens} 가 주 결과다. 셔플은 판단 시각마다 종목 간 점수를 무작위로 섞은 뒤 같은 코호트 규칙으로 돌린 값이며 폴드마다 두 번이다.")
erows = []
for _, r in ens.iterrows():
    w = {"2024-06~08": "A 2024-06~08", "2024-12~2025-02": "B 2024-12~2025-02", "2025-09~11": "C 2025-09~11", "2025-12~2026-02": "D 2025-12~2026-02", "pooled_408d": "합산 408일", "holdout_2026-03~05": "홀드아웃 2026-03~05"}[r.window]
    g = {"A 2024-06~08": "4.48", "B 2024-12~2025-02": "2.68", "C 2025-09~11": "1.54", "D 2025-12~2026-02": "1.22", "합산 408일": "2.18 (p 0.0005, MDD −5.7%)", "홀드아웃 2026-03~05": "—"}[w]
    sh = "—" if pd.isna(r.shuffle1) else f"{r.shuffle1:.2f} / {r.shuffle2:.2f}"
    pv = "< 0.0001 (t 4.06)" if r.p < 0.0001 else f"{r.p:.3f}"
    fundmap = {"A 2024-06~08": "A", "B 2024-12~2025-02": "B", "C 2025-09~11": "C", "D 2025-12~2026-02": "D", "합산 408일": "합산", "홀드아웃 2026-03~05": "홀드아웃 03~05"}
    _ft = pd.read_csv(f"{OUTDIR}/funding_table.csv").set_index("fold"); fsh = f"{_ft.loc[fundmap[w], 'sharpe_fund']:.2f}"
    erows.append([(w,) if w.startswith("합산") or w.startswith("홀드") else w, (f"{r.sharpe:.2f}",), pv, f"{r.mdd:.1f}%", f"×{r.final:.2f}", fsh, g, sh])
table("heatf 소형 CNN 시드 10개 앙상블: 코호트 롱숏 (상·하위 10%, 14일, 왕복 0.2%)@@ens", ["시험 구간", "Sharpe", "NW p", "MDD", "순자산", "Sharpe (펀딩 반영)", "LightGBM 같은 구간", "셔플 1 / 2"], erows, widths=[3.0, 1.3, 2.3, 1.3, 1.3, 1.7, 3.2, 1.9], size=8, note="펀딩 반영 열은 실측 펀딩비(바이낸스 8시간 정산)를 같은 코호트에 넣은 값이며 상세는 부록 C. 본문의 판정은 펀딩 반영 전 기준이다.")
para("네 폴드 전부 Sharpe 2 이상이고 합산은 LightGBM 의 2.18 을 넘으며 MDD 도 작다({F:equity}, {F:cnnvsgbm}). 실측 펀딩비를 넣으면 합산 2.95(p 0.0005), 홀드아웃 2.00(p 0.036)으로 낮아지지만 폴드 넷 모두 양수이고 유의성은 유지된다(부록 C). 폴드 하나(3개월)로는 p 가 0.1 근처까지 올라가지만 90일 표본의 검정력 한계이고 합산 408일에서는 t 가 4 를 넘는다. 셔플은 열 번 모두 음수였다. 결과가 우연이라면 실제 값이 셔플 분포 안에 들어와야 하나 그러한 구간은 없었다. 시드 3개(0·1·2)만 쓴 앙상블은 합산 3.01(p 0.0001)이었고 열 개로 늘리자 3.53 으로 올랐다. 약한 폴드 D 도 1.45 에서 2.28 이 됐다.")
figure("fig7_equity.png", "시드 10개 앙상블의 순자산 곡선 (폴드 A~D, 홀드아웃)과 LightGBM·셔플 대조@@equity", 15.5)
figure("fig8_cnn_vs_gbm.png", "폴드별 Sharpe: LightGBM v2, heatf CNN 시드 0, heatf CNN 시드 10 앙상블@@cnnvsgbm", 14.5)
para("월별로 보면 보유 잔여를 포함한 열여덟 달 가운데 열여섯 달이 양수였다({F:monthly}). 손실 달은 폴드 A 의 2024년 7월(−0.2%)과 폴드 D 의 2026년 2월(−0.3%) 둘뿐이고 가장 좋은 달은 폴드 B 의 2025년 1월(+9.6%)이다. 폴드 D 와 홀드아웃은 첫 달이 가장 컸지만 폴드 A~C 에는 그런 경향이 없다.")
figure("fig9_monthly.png", "시드 10개 앙상블의 월별 코호트 롱숏 수익률 (폴드 A~D, 홀드아웃)@@monthly", 14.5)
section("제4절 방어선 3: 확정 뒤 1회 홀드아웃")
para("그림·모델·지평을 네 폴드로 확정한 뒤 2026-03~05 를 한 번만 썼다. 앙상블 Sharpe 2.49(p 0.019, MDD −7.9%, 순자산 ×1.12), 셔플 −5.1·−3.5 이고 시드 열 개 전부 1.3 이상, 여섯 개가 단독으로 p < 0.05 다({T:holdseeds}). 네 폴드가 모두 알트코인 약세 분기였던 것과 달리 홀드아웃은 학습 데이터에 없던 시기이며 달러중립 롱숏이라 시장 방향의 도움을 받지 않는다. 가설 2 는 채택되었다. 이 채택이 말하는 것은 CNN 이라는 모델의 우위가 아니라 그림의 우위다. 같은 CNN 이 캔들 그림에서는 같은 대조군에 전부 졌고 열지도에서만 넘었다. 대조군이 재는 것은 같은 정보를 수치로 받았을 때의 기준선이고 그 선을 넘었다는 것은 표현이 CNN 으로 하여금 정보를 더 잘 쓰게 했다는 증거다.")
hrows = [[str(int(r.seed)), fmt(r["holdout_2026-03~05_sharpe"]), fmt(r["holdout_2026-03~05_p"], 4), f"{r['holdout_2026-03~05_mdd']:.1f}%", "○" if r["holdout_2026-03~05_p"] < 0.05 else ""] for _, r in seed.iterrows()]
hrows.append([("앙상블 10",), ("2.49",), ("0.019",), "−7.9%", "○"])
table("홀드아웃 2026-03~05 시드별 결과@@holdseeds", ["시드", "Sharpe", "NW p", "MDD", "p < 0.05"], hrows, widths=[2.2, 2.2, 2.2, 2.2, 2.2], size=8.5)
section("제5절 가설 3: 방향은 CNN, 크기는 발생 모델")
para("가설 2 의 CNN 은 어느 종목이 다른 종목보다 오를지를 정하지만 얼마나 움직일지는 모른다. 같은 데이터로 학습한 급등·급락 발생 모델(부록 D)은 반대로 방향은 못 맞히지만 큰 움직임이 올 창은 가려낸다. 둘을 한 모델에 합치는 대신 역할을 분리하는 것이 세 번째 가설이다.")
para("가설 3. 방향 예측(heatf CNN)과 크기·변동성 예측(급등·급락 발생 모델)을 결합한 2단계 포트폴리오 비중 배분은 단독 방향 모델보다 통계적으로 유의한 추가 알파를 낸다.", bold=True, indent=False, before=4, after=4)
para("결합 규칙은 다음과 같다. 코호트의 롱·숏 다리는 그대로 CNN 점수의 상·하위 10% 로 고르고 다리 안 비중만 발생 모델의 급등+급락 확률(15분봉 120봉 창, 판단 직전 75분 안의 최신값)을 판단 시각별 백분위 순위로 바꿔 비례 배분한다. 비교를 위해 두 점수를 z-합으로 섞는 판, 지평 변동성의 역수로 비중을 주는 판, 둘을 곱한 판, 확률 상위 절반 안에서만 같은 수의 종목을 고르는 게이트 판을 같은 4폴드·같은 코호트 규칙으로 돌렸다({T:stack}, {F:stack}).")
para("CNN 과 LightGBM 의 방향 점수는 판단 시각별 Spearman 0.47~0.77 로 많이 겹쳐 z-합 50/50 은 합산 3.53 을 3.00 으로 낮췄다. 반면 발생 확률 순위를 다리 안 비중으로 주면 합산 3.87 로 오르고(Δ +0.33, 14일 블록 부트스트랩 95% 신뢰구간 −0.04~0.64, P(Δ≤0) 0.04) 4폴드 중 3폴드가 개선되며 MDD 는 −4.2% 에서 −4.1% 로 같다. 지평 변동성의 역수 비중은 3.86(Δ +0.32, P 0.12), 둘을 곱한 비중은 4.03(Δ +0.49, P 0.09)이었고 게이트는 3.56 으로 차이가 없었다. 방향 신호를 섞는 것은 손해이고 크기만 다른 모델에 맡기는 것이 이득이다. 가설 3 은 단측 5% 에서 채택되며 신뢰구간이 0 을 살짝 품으므로 추가 알파의 크기는 0.3 안팎으로 읽는다.")
stk = pd.read_csv("/home/arcosium/projects/HYFE_QTPA/results/stack_table.csv"); sp = stk[stk.fold == "pooled"].set_index("variant"); sf = stk[stk.fold != "pooled"].pivot(index="variant", columns="fold", values="sharpe")
sname = {"cnn": "CNN 단독(기준)", "gbm": "LightGBM v2 단독", "mix": "CNN·LightGBM z-합 50/50", "wev": "CNN 방향 + 발생 확률 순위 비중", "wiv": "CNN 방향 + 1/σ 비중", "wevi": "CNN 방향 + 발생 확률 × 1/σ 비중", "gate": "발생 확률 상위 절반 게이트(같은 종목 수)"}
srows = []
for k in ("cnn", "gbm", "mix", "wev", "wiv", "wevi", "gate"):
    r = sp.loc[k]; f = sf.loc[k]; bold = k == "wev"
    d = "—" if pd.isna(r.get("dSharpe_vs_cnn", float("nan"))) else f"{r.dSharpe_vs_cnn:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}] · {r.p_le0:.2f}"
    srows.append([(sname[k],) if bold else sname[k], " · ".join(f"{f[c]:.2f}" for c in sorted(f.index, key=int)[::-1]), (f"{r.sharpe:.2f}",) if bold else f"{r.sharpe:.2f}", "< 0.0001" if r.p < 0.0001 else f"{r.p:.4f}", f"{r.mdd:.1f}%", d])
table("방향은 CNN, 크기는 발생 모델: 다리 안 비중 변형의 코호트 롱숏 성과 (4폴드, 상·하위 10%, 14일, 왕복 0.2%)@@stack", ["구성", "폴드 A·B·C·D Sharpe", "합산 Sharpe", "NW p", "MDD", "Δ vs CNN [95% CI] · P(Δ≤0)"], srows, widths=[5.4, 3.0, 1.5, 1.4, 1.2, 3.6], size=8, note="발생 확률 = 15m·120·10 LightGBM 의 급등+급락 확률, 판단 시각 직전 75분 안 최신값(덮음 99.6% 이상), 판단 시각별 백분위 순위를 비중으로. 신뢰구간은 같은 날을 함께 뽑는 14일 블록 부트스트랩 2,000회.")
figure("fig19_stack.png", "가설 3: 다리 안 비중 변형의 4폴드 합산 Sharpe (막대)와 폴드별 값 (점)@@stack", 14.5)
para("두 신호가 실제로 다른 것을 보는지도 쟀다({T:orth}). 같은 판단 시각 안에서 CNN 방향 점수와 발생 확률의 Spearman 상관은 4폴드 평균 0.07(0.03~0.10)이고 CNN 점수의 절댓값과 발생 확률의 상관은 0.01 이다. 그림이 낸 방향 신호에는 크기 정보가 거의 없고 발생 모델은 방향과 무관한 축을 본다. 발생 모델의 목적은 14일 수익률 예측이 아니라 2.5시간 안의 변동성 사건 확률을 식별하는 것이고 그 일은 부록 D 의 lift 2.3 으로 하고 있다. 그 확률이 14일 상대수익의 절댓값과 상관이 없는 것(−0.03)은 결함이 아니라 모델 목적의 경계에 해당한다. 확률은 종목의 지평 변동성과 반대로 움직이는데(σ(H) 와 −0.14) 변동성이 낮은 종목일수록 k·σ 문턱이 낮아 사건이 잦기 때문이다. 그 결과 발생 확률 비중은 저변동 종목에 자본을 나누는 동적 위험 배분 효과를 주로 내고 변동성 역수 비중이 같은 +0.32 를 내는 것과 맞으며, 둘을 곱했을 때 +0.49 로 더 오르는 만큼이 사건 확률 고유의 정보다. 방향 신호와 위험 배분 신호가 서로 겹치지 않는 채로 결합해 Sharpe 를 올리는 것은 포트폴리오 구성에서 가장 바람직한 구조다. 가설 3 의 결론은 이미지가 방향을 읽고 크기는 독립적인 잣대가 정하며 그 결합이 단측 5% 수준에서 유의한 추가 알파를 낸다는 것이다.")
orth = pd.read_csv("/home/arcosium/projects/HYFE_QTPA/results/ext_orth.csv")
orows = [[(r.fold,) if r.fold == "평균" else r.fold, f"{int(r.n):,}", f"{r.rho_cnn_pev:+.3f}", f"{r.rho_abscnn_pev:+.3f}", f"{r.rho_pev_absfwd:+.3f}", f"{r.rho_pev_sigH:+.3f}", f"{r.rho_cnn_fwd:+.3f}"] for _, r in orth.iterrows()]
table("CNN 방향 점수와 발생 확률의 횡단면 Spearman 상관 (판단 시각별 상관의 평균, 4폴드)@@orth", ["폴드", "창 수", "CNN 점수 · 발생 확률", "|CNN 점수| · 발생 확률", "발생 확률 · |14일 상대수익|", "발생 확률 · σ(H)", "CNN 점수 · 14일 상대수익"], orows, widths=[1.4, 1.6, 2.4, 2.4, 2.8, 2.2, 2.6], size=8, note="발생 확률 = 급등+급락 확률(15m·120·10, 판단 직전 75분 안 최신값). σ(H) = 판단 시각의 지평 변동성 추정. 상관은 판단 시각마다 종목 간 Spearman 을 구해 평균한 값.")

# ================= 제6장 =================
chapter("제6장 논 의")
section("제1절 왜 열지도인가")
para("열지도는 정규화 기준을 창 밖에 두므로 그림 자체에 맥락이 들어가고 CNN 은 그 위에서 봉별 상호작용을 학습한다. 요약 피처 행이 큰 몫을 하지만 {T:ladder} 의 heatx 가 보여 주듯 봉별 순위 행만으로도 부호가 바뀐다. 같은 피처를 표로 받은 LightGBM 보다, 그리고 같은 수치 행렬을 평평하게 받은 MLP(제4장 제5절)보다 heatf CNN 이 낫다는 점에서 2D 합성곱이 실제로 기여한다. 순위 지표인 spread_z 는 CNN 이 오히려 낮은 편인데(시드 0 기준 0.13·0.30·0.21 대 LightGBM 0.26·0.35·0.36) 코호트 손익은 CNN 이 더 좋다. 순위의 평균 품질보다 십분위 양 끝에서 어떤 종목을 고르느냐가 달라 CNN 이 꼬리 종목을 덜 잡기 때문이다.")
para("어느 행이 알파를 담는지는 최종 모델을 대상으로 확인하였다({T:occl}, {F:sal}). 홀드아웃 2026-03~05 의 창 5,000개에서 행군을 표본 평균 그림으로 가리고(Zeiler·Fergus 2014) 원 점수와의 Spearman 순위 상관을 재면 횡단면 백분위 피처 10행을 가릴 때 0.62, 창 끝 피처 27행 0.76, 순위 3행 0.94, 가격 4행 0.97, 거래량 2행 0.99 로 내려간다. 행 하나씩 가리면 24시간 범위의 횡단면 순위, Amihud 비유동성, 30일 대비 거래대금, 상위 5봉 거래 집중도, 급등 횟수처럼 범위·유동성·거래 집중도 행이 앞에 서고 O·H·L·C 각 행은 0.99 이상이다. 기울기 saliency(Simonyan 외 2014)도 같은 행에 몰린다. 시간 방향으로는 피처 띠가 가로줄이라 평평하고 가격 행만 창 앞쪽(첫 10봉 |Δ|/σ 0.09)이 판단 직전(0.02)보다 커서 CNN 이 가격 행에서 읽는 것은 10일 전 대비 위치, 곧 모멘텀이다. 알파의 대부분은 창 밖 정규화된 맥락 행에서 나오고 시계열 열지도는 작지만 0 이 아닌 몫을 더한다. 가격 4행을 가려도 상위 십분위의 86%, 하위의 78% 가 유지되는 것이 그 크기다.")
occ = pd.read_csv("/home/arcosium/projects/HYFE_QTPA/results/occlusion_table_mean.csv"); occz = pd.read_csv("/home/arcosium/projects/HYFE_QTPA/results/occlusion_table_zero.csv").set_index("mask")
_gn = {"xs10": "횡단면 백분위 피처 10행", "feat27": "창 끝 피처 27행", "rank": "봉별 횡단면 순위 3행", "price": "가격 O·H·L·C 4행", "vol": "거래량 2행", "time": "시각·요일 2행"}
orows = []
for k in ("xs10", "feat27", "rank", "price", "vol", "time"):
    r = occ[occ["mask"] == k].iloc[0]; orows.append([(_gn[k],) if k in ("xs10", "feat27") else _gn[k], f"{r['rows']}", (f"{r.spearman:.2f}",) if k in ("xs10", "feat27") else f"{r.spearman:.2f}", f"{occz.at[k, 'spearman']:.2f}", f"{r.abs_delta_over_sd:.2f}", f"{r.top_kept*100:.0f}%", f"{r.bot_kept*100:.0f}%"])
_single = occ[occ["rows"] == 1].sort_values("spearman").head(5)
for _, r in _single.iterrows():
    orows.append([f"행 하나: {r['mask'].split(':')[1]}", "1", f"{r.spearman:.2f}", f"{occz.at[r['mask'], 'spearman']:.2f}", f"{r.abs_delta_over_sd:.2f}", f"{r.top_kept*100:.0f}%", f"{r.bot_kept*100:.0f}%"])
table("행군을 가렸을 때 점수의 변화 (홀드아웃 2026-03~05 창 5,000개, 최종 모델 시드 0~2 평균)@@occl", ["가린 행", "행 수", "Spearman (평균 그림 가림)", "Spearman (0 가림)", "|Δ점수|/σ", "상위 십분위 유지", "하위 십분위 유지"], orows, widths=[4.6, 1.0, 2.4, 2.0, 1.6, 2.0, 2.0], size=8, note="평균 그림 가림 = 가린 픽셀을 표본 5,000개의 평균값으로 채움(분포 안). 0 가림 = 0 으로 채움(피처 행에는 극단값이라 변화가 과대). 유지 = 가린 뒤에도 같은 십분위에 남는 비율. 시각·요일 행은 학습 렌더에서 0 에 가까운 값이라 가려도 변화가 없다.")
figure("fig14_saliency.png", "최종 모델이 보는 곳: 기울기 saliency 지도(왼쪽, 진할수록 큼)와 행군 가림 시 순위 상관 손실(오른쪽)@@sal", 15.0)
para("가림은 이미 학습된 모델이 무엇에 의존하는지를 보이지만 그 행이 없어도 되는지는 알려주지 않으므로 이미지를 바꾸어 처음부터 재학습하였다({T:layout}). 48행의 순서를 무작위로 고정해 섞은 판(행 순열, 시드마다 다른 순열), 60봉의 시간 순서를 섞은 판(봉 순열), 행군 하나를 0 으로 채운 판(제거)을 같은 시드·같은 폴드·같은 코호트 규칙으로 돌렸다. 행 순열은 시드 평균 2.34, 3시드 앙상블 2.48 로 기준(2.56, 3.01)보다 낮지만 무너지지 않는다. 행의 이웃 관계는 도움이 되되 필수가 아니다. 봉 순열은 2.75, 3.16 으로 기준과 같거나 높다. 봉의 국소 순서, 곧 캔들 패턴 같은 형태는 이 알파에 쓰이지 않는다는 뜻이며 제3장의 진단과 맞는다. 행군을 빼면 피처 27행 제거가 2.13(앙상블 2.18)으로 가장 크게 떨어지고 가격 4행 2.30(2.45), 순위 3행 2.33(2.53), 횡단면 피처 10행 2.30(2.58), 거래량 2행 2.50(2.74), 시각·요일 2행 3.01 순이다. 가림에서 가장 컸던 횡단면 10행이 재학습에서는 작게 나오는 것은 같은 정보가 피처 27행과 순위 3행에도 들어 있어 모델이 다른 행에서 다시 배우기 때문이다. 어느 행군 하나도 필수는 아니고 알파는 여러 맥락 행에 나뉘어 있으며 그 가운데 창 끝 피처 띠의 몫이 가장 크다. 시드 두세 개의 실험이라 ±0.3 안의 차이는 시드 편차(합산 표준편차 0.4) 안에 있고 피처 27행 제거만 기준의 최소 시드(2.29)보다 낮은 1.60 을 냈다.")
lay = pd.read_csv("/home/arcosium/projects/HYFE_QTPA/results/ext_layout.csv")
lrows2 = []
for _, r in lay.iterrows():
    if r.n_seeds == 0: continue
    b = r.variant.startswith("기준")
    lrows2.append([(r.variant,) if b else r.variant, f"{int(r.n_seeds)}", fmt(r.seed_pooled_mean), fmt(r.seed_pooled_min), (fmt(r.ens_sharpe),) if b else fmt(r.ens_sharpe), fmt(r.ens_p, 4), "—" if pd.isna(r.ens_mdd) else f"{r.ens_mdd:.1f}%", "—" if pd.isna(r.ens_folds) else str(r.ens_folds)])
table("그림을 바꿔 다시 학습한 heatf 소형 CNN 의 코호트 롱숏 (4폴드, 상·하위 10%, 14일, 왕복 0.2%)@@layout", ["그림", "시드 수", "시드별 합산 평균", "시드별 합산 최소", "시드 앙상블 합산", "NW p", "MDD", "앙상블 폴드 A·B·C·D"], lrows2, widths=[3.8, 1.1, 1.7, 1.7, 1.7, 1.4, 1.2, 3.4], size=8, note="행 순열·봉 순열은 시드 0·1·2(순열 시드 1·2·3), 행군 제거는 시드 0·1. 기준은 같은 시드 수의 앙상블. 제거 = 해당 행을 0 으로 채운 그림으로 학습. 시각·요일 2행은 학습 렌더에서 0 에 가까워 제거해도 차이가 없는 대조.")
section("제2절 파라미터 민감도: 왜 14일과 4시간봉인가")
para("14일 지평과 4시간봉의 선택 근거는 파라미터 민감도 분석으로 제시한다({T:horizon}). 7일(42봉)·21일(126봉)·28일(168봉) 지평으로 라벨부터 다시 학습해 시드 0~2 의 앙상블을 만들면 4폴드 합산 Sharpe 는 2.00·2.28·2.75 로 전부 2 안팎이고 합산 p 는 0.003 이하이며 확정 뒤 홀드아웃에서도 2.10(p 0.069)·1.80(p 0.044)·2.49(p 0.024)로 부호가 유지된다. 격자를 1시간봉으로 바꾸고 창(10일)과 지평(14일)을 달력 기준으로 같게 두면 2.57(p 0.0009)로 4시간봉의 2.81 과 비슷하다. 즉 어느 지평·격자에서도 신호는 유의하고 그 가운데 14일·4시간봉이 합산 3.01, 홀드아웃 3.30 으로 가장 좋다. 이 조합은 백테스트의 우연이 아니라 민감도 분석이 가리키는 정점이며 4시간봉 자체보다 창과 지평의 길이가 성과를 결정한다. 보유 기간만 바꿔 본 반감기 곡선(부록 E)도 같은 정점을 가리킨다.")
hz = pd.read_csv("/home/arcosium/projects/HYFE_QTPA/results/ext_horizon.csv"); g1 = pd.read_csv("/home/arcosium/projects/HYFE_QTPA/results/ext_grid1h.csv")
hrows2 = []
for _, r in hz.iterrows():
    b = "기준" in r.variant
    hrows2.append([(r.variant,) if b else r.variant, f"{int(r.n_seeds)}", fmt(r.seed_pooled_mean), fmt(r.seed_pooled_min), (fmt(r.ens_sharpe),) if b else fmt(r.ens_sharpe), fmt(r.ens_p, 4), "—" if pd.isna(r.ens_mdd) else f"{r.ens_mdd:.1f}%", (fmt(r.get("hold_ens_sharpe")),) if b else fmt(r.get("hold_ens_sharpe")), fmt(r.get("hold_ens_p"), 4)])
for _, r in g1.iterrows():
    if r.variant.startswith("1h"):
        hrows2.append(["1h·240·336 (같은 10일 창·14일 지평, 판단 4시간마다)", f"{int(r.n_seeds)}", fmt(r.seed_pooled_mean), fmt(r.seed_pooled_min), fmt(r.ens_sharpe), fmt(r.ens_p, 4), "—" if pd.isna(r.ens_mdd) else f"{r.ens_mdd:.1f}%", "—", "—"])
table("지평·격자를 바꿔 다시 학습한 heatf 소형 CNN (4폴드 합산과 홀드아웃 2026-03~05, 코호트 롱숏)@@horizon", ["지평", "시드 수", "시드별 합산 평균", "시드별 합산 최소", "시드 앙상블 합산", "NW p", "MDD", "홀드아웃 앙상블", "홀드아웃 p"], hrows2, widths=[4.2, 1.0, 1.6, 1.6, 1.6, 1.3, 1.2, 1.7, 1.4], size=8, note="지평별 시드 0~2, 보유 봉수 = 지평. 기준(14일)은 같은 시드 0~2 의 앙상블이라 본문의 시드 10개 앙상블(3.53, 홀드아웃 2.49)과 다르다. 1시간봉 판(시드 0·1)은 판단 시각을 4시간마다로 맞춰 같은 수의 코호트를 연다.")
section("제3절 실무 함의")
para("판정은 실전과 같은 규칙(판단 시각별 십분위, 14일 보유, 왕복 0.2%)으로 했고 유니버스는 판단 시점 유동성으로만 걸렀다. 이 연구의 모든 판정은 학습 종료 뒤 1~3개월 구간에서 이루어졌으므로 실전에서도 분기 단위 재학습을 전제로 한다. 폴드 D 앙상블의 월 수익이 +5.6%·+1.3%·−0.3%, 홀드아웃이 +6.0%·+4.3%·+1.3% 로 첫 달이 가장 컸다는 점도 학습 시점과의 거리를 짧게 유지해야 할 이유다. 부록 A 의 페이퍼 장부는 이 모델로 실제 봉에서 신호를 내는 구현이 가능함을 보이려는 것이고 운용 성과 측정이 목적은 아니다.")
para("시장 노출이 없는 롱숏의 성격은 같은 기간의 BTC 단순 보유와 비교하면 드러난다({T:btc}, {F:btc}). 네 폴드를 이어 붙인 408일 동안 BTC 는 −54.8%(최대낙폭 −59.7%)였고 앙상블 롱숏은 +62.6%(−4.2%)였다. 네 폴드가 모두 알트코인 약세 분기였으므로 롱숏이 시장 방향과 무관하게 벌었다는 뜻이다. 홀드아웃 2026-03~05 는 BTC 가 +10.0%(−11.8%)로 올랐고 롱숏은 +12.0%(−7.9%)였다. 절대 수익은 BTC 상승기에 비슷하고 하락기에 크게 앞서며 낙폭은 어느 구간에서든 BTC 의 10분의 1 에서 3분의 2 다. 이 비교는 위험이 다른 두 자산의 나열이지 같은 위험에서의 우열이 아니다. 연율 변동성이 롱숏은 구간마다 7~19%, BTC 는 39~56% 라 위험 단위당 수익으로 보면 격차는 더 벌어진다.")
bc = pd.read_csv(f"{OUTDIR}/btc_compare.csv")
brows = [[(r.period,) if ("합산" in r.period or "홀드" in r.period) else r.period, f"{int(r.days)}", (f"{r.strat_ret:+.1f}%",) if ("합산" in r.period or "홀드" in r.period) else f"{r.strat_ret:+.1f}%", f"{r.strat_mdd:.1f}%", f"{r.strat_vol:.1f}%", f"{r.btc_ret:+.1f}%", f"{r.btc_mdd:.1f}%", f"{r.btc_vol:.1f}%"] for _, r in bc.iterrows()]
table("heatf 롱숏(시드 10 앙상블)과 BTC 단순 보유의 구간별 수익·최대낙폭·연율 변동성@@btc", ["구간", "일수", "롱숏 수익", "롱숏 MDD", "롱숏 변동성", "BTC 수익", "BTC MDD", "BTC 변동성"], brows, widths=[3.4, 1.2, 1.8, 1.8, 1.9, 1.8, 1.8, 1.9], size=8, note="롱숏 = 코호트 롱숏(상·하위 10%, 14일, 왕복 0.2%, 펀딩 전), BTC = 같은 날짜의 4시간봉 일별 종가 단순 보유. 합산은 네 폴드를 이어 붙인 것이라 구간 사이가 불연속이다.")
figure("fig15_btc.png", "heatf 롱숏과 BTC 단순 보유의 순자산: 롤링 4폴드 이어붙임(왼쪽)과 홀드아웃(오른쪽)@@btc", 15.0)
para("운용 가능 규모는 제4절의 충격비용 분석에, 시장 국면별 성과와 보유 기간별 곡선은 부록 E 에 제시한다.")
section("제4절 한계")
para("시험 구간이 3개월씩이라 폴드 하나의 검정력이 낮고 유의성은 합산과 홀드아웃에 기댄다. 네 폴드는 모두 알트코인 약세 분기였고 학습 데이터에 없던 시기의 표본은 홀드아웃 하나뿐이다. 캔들 25개 설정은 모두 4폴드를 돌렸지만 같은 폴드·같은 유니버스를 쓰므로 서로 독립이 아니고 기각의 근거는 설정 수보다 폴드 값의 범위(−0.22~+0.16)와 적중률(설정 평균 41~48%)에 있다. 유동성 상위 200 종목 밖과 4시간 이외 격자에는 결과를 일반화하지 않는다.")
para("거래비용은 더 큰 한계다. 기본 판정의 왕복 0.2% 는 고정 가정이고 펀딩비는 {T:ens} 에 병기하고 부록 C 에서 따로 반영했다. 펀딩 자료가 유니버스의 절반 안팎만 덮고 자료 없는 종목에 같은 다리 평균을 적용했으므로 실제 펀딩 부담은 표보다 클 수 있다. 슬리피지와 미체결은 호가 자료가 없어 재지 못했고 그 대신 두 다리 종목의 유동성을 {T:liq} 에 적는다. 롱 다리 종목의 일평균 달러 거래대금 중앙값은 380만~2,800만 달러, 하위 10% 는 170만~540만 달러로 숏 다리(중앙값 3,200만~6,700만 달러)보다 얇다. 상위 십분위에 유동성이 낮은 종목이 몰리기 때문이다. 종목당 포지션을 일거래대금의 1% 이하로 두어야 0.2% 가정이 성립한다고 보면, 롱 다리 하위 10% 종목 기준 종목당 1.7만~5.4만 달러, 코호트당(롱 20종목) 34만~108만 달러이고 84개 코호트가 겹치는 구조에서 전체 운용 규모는 수천만 달러 수준이 상한이다. 아래의 제곱근 충격 모형({T:cap})도 같은 답을 준다. 그 이상에서는 비용이 0.2% 를 넘고 하위 십분위 공매도는 대차 가능 여부와 청산 위험까지 더해진다. 그럼에도 실측 펀딩비를 직접 차감한 검증(부록 C)에서도 합산 Sharpe 2.95(p 0.0005)로 유의성이 유지됨을 확인하였다.")
liq = pd.read_csv(f"{OUTDIR}/liquidity_legs.csv")
lrows = [[r.fold, f"{r.long_med:.1f}", f"{r.long_p10:.1f}", f"{r.short_med:.1f}", f"{r.short_p10:.1f}"] for _, r in liq.iterrows()]
table("코호트 롱·숏 다리 종목의 유동성 (판단 시점 직전 30일 일평균 달러 거래대금, 백만 달러)@@liq", ["시험 구간", "롱 다리 중앙값", "롱 다리 하위 10%", "숏 다리 중앙값", "숏 다리 하위 10%"], lrows, widths=[3.4, 2.6, 2.6, 2.6, 2.6], size=8.5, note="시드 10개 앙상블의 상·하위 십분위 종목, 1시간봉 거래대금 합. 롱 다리가 숏 다리보다 얇다.")
para("운용 가능 규모는 제곱근 충격 모형(Tóth 외 2011)으로 추정하였다({T:cap}). 종목당 하루 편도 거래 명목 Q 를 운용 규모 F 로부터 F·6/(2·84·20)으로 두고(4시간마다 코호트 하나가 열리고 닫히며 84개가 겹친다) 편도 충격을 0.7·σ_일·√(Q/ADV)로 계산해 왕복 0.2% 에 더한 뒤 같은 코호트를 다시 판정했다. ADV 는 판단 직전 30일 일평균 달러 거래대금이고 두 다리 종목의 중앙값은 폴드별 1,850만~5,690만 달러, 하위 10% 는 210만~760만 달러다. 운용 규모 100만 달러에서 충격은 편도 4bp 로 합산 Sharpe 3.37, 1,000만 달러 13bp 에 3.02, 3,000만 달러 22bp 에 2.64, 1억 달러 40bp 에 1.90, 3억 달러 69bp 에 0.70 이고 10억 달러에서는 음수다. 실용적인 상한은 수천만 달러이며 위의 유동성 어림과 같다. 이 전략은 수천만 달러(약 100~300억 원) 규모에 맞는 알파이고 그 이상의 자금에서는 거래대금이 얇은 알트코인의 충격비용과 슬리피지가 알파를 잠식한다. 대형 운용사의 알파와 중소형 퀀트 펀드의 알파는 다를 수밖에 없다.")
cap = pd.read_csv("/home/arcosium/projects/HYFE_QTPA/results/ext_capacity.csv")
caprows = [[f"{int(r.fund_usd_m):,}", f"{r.impact_bp_side:.1f}", (f"{r.sharpe:.2f}",) if r.fund_usd_m == 10 else f"{r.sharpe:.2f}", fmt(r.p, 4), f"{r.mdd:.1f}%", f"×{r.final:.2f}"] for _, r in cap.iterrows()]
table("운용 규모별 충격비용과 코호트 롱숏 성과 (제곱근 충격, Y=0.7, 시드 10 앙상블 4폴드 합산)@@cap", ["운용 규모 (백만 달러)", "편도 충격 (bp, 다리 평균)", "합산 Sharpe", "NW p", "MDD", "순자산"], caprows, widths=[2.8, 2.8, 2.2, 2.0, 1.8, 1.8], size=8.5, note="충격 = 0.7·σ_일·√(Q/ADV), Q = 종목당 하루 편도 명목 = F·6/(2·84·20), ADV = 판단 직전 30일 일평균 달러 거래대금(4시간봉 거래대금 합). 왕복 0.2% 에 편도 충격을 더해 재판정. 호가 자료 없이 모형으로 어림한 값이다.")
para("공매도 다리의 현실 제약도 짚어 둔다. 이 전략은 무기한선물이므로 현물 공매도의 대차 수수료는 없고 그 자리를 펀딩비가 대신하며 그 부담은 부록 C 에 실측으로 넣었다(숏 다리가 폴드 C·D 에서 14일 보유당 155~200bp 를 냄). 재지 못한 것은 청산 위험과 거래소 포지션 한도, 얇은 호가의 슬리피지, 상장폐지 시 강제 정산이다. 이 가운데 청산 위험은 코호트 자료로 가늠할 수 있다. 네 폴드 숏 다리 포지션 38,667개 가운데 14일 만기에 50% 넘게 오른 종목은 1.2%, 두 배 넘게 오른 종목은 0.3% 였고 보유 중 고점 기준으로는 3.7% 와 0.9% 였다. 가장 큰 사례는 폴드 C 의 +429% 다. 다리 안 동일비중(20 종목, 종목당 자본의 2.5%)과 명목 1배에서는 한 종목이 두 배가 되어도 자본의 2.5% 손실이고 +429% 사례도 10.7% 라 계좌 청산은 일어나지 않는다. 이 손실은 이미 단순수익 코호트 판정과 MDD −4.2% 안에 들어 있다. 청산 위험은 레버리지를 쓸 때 생기며 그때는 스퀴즈가 잦은 폴드 C(보유 중 두 배 1.9%)가 병목이다. 숏 다리의 평균 만기 수익이 −9.2% 로 롱 다리의 −5.9% 보다 좋다는 점에서 이 전략의 알파 상당 부분은 숏 다리에서 나오므로 공매도가 막히면(거래소 제한, 상장폐지) 성과는 롱 다리 초과수익 수준으로 내려간다.")
para("같은 그림을 KRX 일봉에 옮긴 예비 실험이 세 시드 모두 음수였던 것은 시장 구조의 차이로 읽힌다. 첫째, 암호화폐는 24시간 연속시장이라 4시간봉과 시각·요일 행이 뜻을 갖지만 KRX 는 6.5시간 장과 장마감 갭이 있어 일봉만 남고 이 행들은 무의미해진다. 둘째, 상하한가 30% 가 꼬리를 자르므로 상대수익 라벨과 k·σ 사건의 분포가 달라진다. 셋째, 공매도 제약으로 하위 십분위를 팔 수 없어 롱온리 초과수익(왕복 0.4%)으로만 판정했고 알파의 큰 몫이 숏 다리에 있는 이 전략에 불리하다. 넷째, 무기한선물의 펀딩·레버리지가 만드는 횡단면 압력이 KRX 에는 없다. 다섯째, 자료가 2024-05 이후라 폴드별 학습 기간이 1년 안팎이고 폴드 하나는 비어 있었다. 어느 요인이 결정적인지는 장중 봉으로 다시 그려 보기 전에는 가를 수 없다.")
# ================= 제7장 =================
chapter("제7장 결 론")
para("본 연구는 주식에서 유효했던 캔들 차트 이미지가 암호화폐 횡단면에서도 유효한가를 검정하였고 실험 결과 가설 0 은 기각되었다. 격자·창·채널·거래량·추세·해상도·종목 수·학습률·유니버스를 바꾼 25개 설정 100개 폴드에서 캔들 이미지 CNN 의 상대방향 점수 차는 −0.22~+0.16 에 머물렀고 사전 등록 기준을 넘긴 설정은 없었다. 실패는 우연이 아니라 구조적 원인에 기인하였다. 모델은 형태 대신 창 안 모멘텀과 수익률 왜도를 외웠고 그 부호는 분기마다 뒤집혔다. 창 안 최고·최저로 정규화된 그림에는 이 종목이 유니버스 안에서 어디에 있는지, 이 거래량이 평소보다 많은지, 지금이 어느 시각인지가 없었다.")
para("원인 둘을 뒤집어 정규화 기준을 창 밖에 둔 열지도 이미지 heatf 를 설계하자 같은 75만 파라미터 소형 CNN 이 교사 없이 같은 정보를 수치로 받은 LightGBM 대조군을 넘었다. 순위 세 행만 더해도 부호가 양수로 돌아섰고 요약 피처 37행까지 더하자 시드 열 개 전부의 앙상블이 4폴드 합산 Sharpe 3.53(p < 0.0001, MDD −4.2%), 확정 뒤 홀드아웃 2.49(p 0.019)를 냈으며 점수 셔플은 모든 구간에서 음수였다. 방향은 CNN 에 맡기고 다리 안 비중을 급등·급락 발생 확률로 주는 2단계 결합(가설 3)은 합산 3.87 로 유의한 추가 알파를 냈다. 같은 수치 행렬을 평평하게 받은 MLP 가 2.05 에 그친 것은 2D 합성곱이 봉과 행의 국소 상호작용을 잡는 몫이 있다는 뜻이고, 가림 실험에서 횡단면 백분위 10행을 가리면 순위 상관이 0.62 로 내려가고 가격 4행을 가리면 0.97 에 머문 것은 알파의 대부분이 창 밖 정규화된 맥락 행에서 나오고 시계열 열지도는 작지만 0 이 아닌 몫을 더한다는 뜻이다. 우위는 CNN 이 아니라 그림에 딸린 것이다. 같은 CNN 이 캔들 그림에서는 같은 대조군에 전부 졌고 열지도에서만 넘었다. LightGBM 은 경쟁자가 아니라 같은 정보를 수치로 받았을 때의 기준선이며 그 선을 넘었다는 것이 표현이 정보를 살렸다는 증거다. 성과를 가르는 요인은 CNN 의 채택 여부가 아니라 CNN 에 제시하는 이미지의 설계이다.")
para("결과의 신뢰성은 평가 규약이 뒷받침한다. 시드 열 개를 미리 정해 전부 보고했고 최고 시드를 고른 표는 없다. 실측 펀딩비를 넣어도 합산 2.95(p 0.0005), 홀드아웃 2.00(p 0.036)으로 유의성이 남는다. 선택은 검증 구간에서만, 판정은 시험 구간에서만 했고 학습·검증·시험을 분기 단위로 굴린 롤링 폴드에 확정 뒤 한 번만 쓴 홀드아웃과 점수 셔플 대조를 더했다. 그럼에도 부록 A 의 순방향 페이퍼 장부가 몇 달을 더 돌아야 이 주장은 완성된다.")
para("실무적 함의는 네 가지다. 첫째, 신호의 실현은 느리다. 지평·격자를 바꿔 다시 학습해도 전부 유의하되 14일·4시간봉이 정점이고 보유 기간만 바꾸면 6주까지 완만하게 식으므로 잦은 회전은 손해다. 둘째, 성과는 시장 방향과 무관하다. 네 폴드 동안 BTC 가 −55% 일 때 롱숏은 +63% 였고 낙폭은 BTC 의 10분의 1 이었다. 셋째, 포지션 크기는 독립적인 잣대가 결정한다. 방향은 CNN 이 정하고 다리 안 비중을 급등·급락 발생 확률로 주면 합산 3.87 로 오르며 두 신호의 상관은 0.07 로 서로 겹치지 않는다(가설 3). 넷째, 운용 용량은 유한하다. 제곱근 충격 모형으로 어림하면 운용 규모 천만 달러까지 Sharpe 3 이 유지되고 1억 달러에서 1.9, 3억 달러에서 0.7 로 내려간다.")
para("향후 과제는 셋이다. 첫째, 이 그림이 다른 시장에서도 통하는지다. 같은 heatf 를 KRX 일봉에 적용한 예비 실험은 세 시드 모두 음수였고 그 구조적 원인은 제6장 제4절에 정리하였으나 어느 요인이 결정적인지는 장중 봉으로 다시 그려 보아야 가릴 수 있다. 둘째, 더 긴 역사다. 데이터가 2023년부터라 강세장 한 구간과 약세장 네 구간뿐이고 2020~2022년의 급등·붕괴 구간은 없다. 셋째, 체결이다. 호가 자료가 없어 슬리피지를 모형으로 대신했고 하위 십분위 공매도의 대차 가능성과 청산 위험은 재지 못했다. 이 셋을 채우기 전까지 이 논문의 주장은 검증 규약 안에서만 성립한다.")

# ================= 참고문헌 =================
chapter("참고문헌")
for ref in ["Amihud, Y. (2002). Illiquidity and Stock Returns: Cross-Section and Time-Series Effects. Journal of Financial Markets, 5(1), 31-56.",
            "Gu, S., Kelly, B., Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. Review of Financial Studies, 33(5), 2223-2273.",
            "Harvey, C. R., Liu, Y., Zhu, H. (2016). ... and the Cross-Section of Expected Returns. Review of Financial Studies, 29(1), 5-68.",
            "He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR, 770-778.",
            "Hou, K., Xue, C., Zhang, L. (2020). Replicating Anomalies. Review of Financial Studies, 33(5), 2019-2133.",
            "Jegadeesh, N., Titman, S. (1993). Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency. Journal of Finance, 48(1), 65-91.",
            "Jiang, J., Kelly, B., Xiu, D. (2023). (Re-)Imag(in)ing Price Trends. Journal of Finance, 78(6), 3193-3249. SSRN 3756587.",
            "Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., Liu, T.-Y. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS 30.",
            "Krauss, C., Do, X. A., Huck, N. (2017). Deep Neural Networks, Gradient-Boosted Trees, Random Forests: Statistical Arbitrage on the S&P 500. European Journal of Operational Research, 259(2), 689-702.",
            "Liu, Y., Tsyvinski, A., Wu, X. (2022). Common Risk Factors in Cryptocurrency. Journal of Finance, 77(2), 1133-1177.",
            "López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.",
            "Newey, W. K., West, K. D. (1987). A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. Econometrica, 55(3), 703-708.",
            "Sezer, O. B., Ozbayoglu, A. M. (2018). Algorithmic Financial Trading with Deep Convolutional Neural Networks: Time Series to Image Conversion Approach. Applied Soft Computing, 70, 525-538.",
            "Simonyan, K., Vedaldi, A., Zisserman, A. (2014). Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps. ICLR Workshop.",
            "Tóth, B., Lempérière, Y., Deremble, C., de Lataillade, J., Kockelkoren, J., Bouchaud, J.-P. (2011). Anomalous Price Impact and the Critical Nature of Liquidity in Financial Markets. Physical Review X, 1(2), 021006.",
            "Wang, Z., Oates, T. (2015). Imaging Time-Series to Improve Classification and Imputation. IJCAI, 3939-3945.",
            "Zeiler, M. D., Fergus, R. (2014). Visualizing and Understanding Convolutional Networks. ECCV, 818-833.",
            "Visualizing Price Trends in China: A Multi-Channel Grayscale CNN Approach (2025). SSRN 5136032.",
            "Visual Chart Representations for Cryptocurrency Regime Prediction (2026). arXiv 2605.00875."]:
    p = doc.add_paragraph(); p.paragraph_format.left_indent = Cm(0.8); p.paragraph_format.first_line_indent = Cm(-0.8); p.paragraph_format.space_after = Pt(4); set_font(p.add_run(ref), 10)

# ================= 부록 =================
chapter("부록 A 실전 페이퍼 장부")
para("4시간봉 마감마다 유동성 상위 약 200 종목의 heatf 이미지를 실시간으로 그려 최종 모델(학습 2023-01~2026-05)의 시드 10개 앙상블 점수로 상·하위 10% 를 14일 보유하는 페이퍼 장부를 가동한다. 실시간 렌더는 학습 렌더와 픽셀 단위로 일치함을 확인했다(네 창 최대 차이 0, 점수 동일). 장부는 구현 가능성을 보이는 부록이며 운용 성과는 이 논문의 주장에 쓰지 않는다.")
para("장부는 2026-09-08 13시(KST)에 첫 판단을 내렸고 4시간마다 유동성 상위 약 200 종목을 그려 상·하위 10% 를 새로 열며 14일 뒤 닫는다({F:live}). 첫 틱에 롱 20·숏 20 을 열었고 이틀 뒤 보유는 90 종목에 이르렀다. 화면({F:shot})은 보유 종목과 평가손익, 선택 종목의 차트, 계좌 요약과 마지막 판단의 상·하위 점수, 급등·급락 경보를 한 창에 보인다. 초기 자본 1만 달러 표기는 포지션 비중(자본의 1%)을 달러로 읽기 위한 것이다.")
figure("fig16_live.png", "실전 페이퍼 장부의 틱별 신규 포지션과 보유 종목 수 (2026-09-08 13시 ~ 09-10 13시 KST)@@live", 14.5)
figure("autocrypto_shot.png", "실전 페이퍼 장부 화면 (AutoCrypto, 2026-09-10 14:52 KST): 보유·평가손익(왼쪽), 선택 종목 1시간봉 차트(가운데), 계좌·마지막 판단·급등급락 경보(오른쪽)@@shot", 15.5)
chapter("부록 B 재현 절차와 결과 원자료")
para("{F:pipe} 이 자료에서 실전 장부까지의 여덟 단계와 각 단계의 코드다. 모든 단계는 저장소의 스크립트 하나로 다시 돌릴 수 있고 학습만 GPU 가 필요하다.")
figure("fig17_pipeline.png", "재현 파이프라인: 1분봉 저장소에서 실전 장부까지@@pipe", 15.5)
bullet("이미지 렌더·학습: hyfe/train_cnn.py --render heatf. 폴드 학습 RENDER=heatf SEED=n bash hyfe/fulldir.sh i1 4h:60:84, 홀드아웃 SEED=n bash hyfe/hold.sh")
bullet("판정: hyfe/cohort.py(코호트 롱숏), hyfe/perf.py(Sharpe·NW p·MDD), 시드 집계 hyfe/eval_seeds.sh")
bullet("대조군: hyfe/pilot_gbm.py --xs --preset small|reg, hyfe/zavg.py (z-평균), 분기 walk-forward --quarters")
bullet("실전: hyfe/live_cnn.py, hyfe/live.py")
bullet("결과 원자료: 저장소 results/ (시드·앙상블 코호트 json, 대조군 시뮬 json, 표 CSV). 실험 전 과정은 실험일지.md")
chapter("부록 C 펀딩비 반영 상세")
para("본문의 성과는 펀딩비를 넣기 전 기준이고 제5장 {T:ens} 에 펀딩 반영 Sharpe 를 병기했다. 이 부록은 그 계산의 상세다. 알트코인 롱숏, 특히 하위 십분위 공매도는 펀딩비를 내는 쪽에 서기 쉽다. 시드 10개 앙상블의 같은 코호트에 실측 펀딩비를 넣은 결과가 {T:fund} 과 {F:fundfig} 다. 폴드 C·D 에서는 숏 다리가 14일 보유당 155~200bp 를 냈고 롱 다리는 오히려 17~99bp 를 받았다. 하락 종목에 공매도가 몰려 펀딩이 음수로 돌아선 구간이다. 폴드 A·B 는 숏 다리가 10~13bp 를 받고 롱 다리가 6~18bp 를 내는 정도라 영향이 작았다. 합산 Sharpe 는 3.53 에서 2.95(p 0.0005, MDD −5.3%, 순자산 ×1.51)로, 홀드아웃은 2.49 에서 2.00(p 0.036, MDD −8.0%)으로 낮아지지만 폴드 넷 모두 양수이고 합산·홀드아웃의 유의성은 유지된다. 펀딩은 실전 성과를 15~40% 깎는 실질 비용이며 이 논문의 목표(구간별 1 초과, 합산 1.5 초과)는 펀딩을 넣어도 충족한다.")
fund = pd.read_csv(f"{OUTDIR}/funding_table.csv")
frows = []
for _, r in fund.iterrows():
    cov = "" if pd.isna(r.covered_bases) else f"{int(r.covered_bases)}"; lp = "" if pd.isna(r.long_paid_bp_per_hold) else f"{-r.long_paid_bp_per_hold:+.0f}"; sp = "" if pd.isna(r.short_received_bp_per_hold) else f"{r.short_received_bp_per_hold:+.0f}"
    name = {"A": "A 2024-06~08", "B": "B 2024-12~2025-02", "C": "C 2025-09~11", "D": "D 2025-12~2026-02", "합산": "합산 408일", "홀드아웃 03~05": "홀드아웃 2026-03~05"}[r.fold]
    bold = r.fold in ("합산", "홀드아웃 03~05")
    gb = "—" if pd.isna(r.gbm_sharpe) else f"{r.gbm_sharpe:.2f} / {r.gbm_sharpe_fund:.2f}"
    frows.append([(name,) if bold else name, f"{r.sharpe:.2f}", (f"{r.sharpe_fund:.2f}",) if bold else f"{r.sharpe_fund:.2f}", f"{r.p_fund:.4f}" if r.p_fund >= 0.0001 else "< 0.0001", f"{r.mdd_fund:.1f}%", lp, sp, cov, gb])
table("펀딩비 반영 전후의 시드 10개 앙상블 성과와 LightGBM 대조 (코호트 롱숏, 왕복 0.2% + 실측 펀딩)@@fund", ["시험 구간", "CNN Sharpe (펀딩 전)", "CNN Sharpe (펀딩 후)", "NW p (후)", "MDD (후)", "롱 다리 펀딩 수취 (bp/14일)", "숏 다리 펀딩 수취 (bp/14일)", "펀딩 자료 종목", "LightGBM Sharpe (전 / 후)"], frows, widths=[2.9, 1.6, 1.6, 1.4, 1.3, 1.9, 1.9, 1.4, 2.3], size=8, note="펀딩 수취는 코호트당 보유 기간(14일) 평균이며 양수가 받는 쪽. 자료 없는 종목은 같은 다리 평균 적용. LightGBM 은 같은 코호트 규칙으로 다시 계산한 값(본문 표의 K=100 슬롯 시뮬과 조금 다르다). 홀드아웃의 LightGBM 은 계산하지 않았다.")
figure("fig18_funding.png", "펀딩비 반영 전후의 폴드별 Sharpe(왼쪽)와 다리별 펀딩 수취(오른쪽)@@fundfig", 15.0)
para("같은 코호트 규칙으로 계산한 LightGBM v2 는 펀딩 전 합산 2.30 에서 펀딩 후 1.99(p 0.007)로 내려가고 폴드 C·D 는 1.03·0.57 이 된다. 펀딩을 넣은 뒤에도 CNN 앙상블(2.95)이 LightGBM(1.99)을 앞선다. 두 모델 모두 같은 십분위 구조라 펀딩 부담은 비슷하게 받고 격차는 유지된다.")

chapter("부록 D 부수 실험: 급등·급락 발생 예측")
para("본문의 방향 예측과 별도로 같은 데이터로 변동성 사건의 발생을 예측하는 실험을 했다. 제3장 제3절이 말한 맥락 부재가 다른 과제에서도 같은 방식으로 나타나는지 보려는 것이다. 15분봉 120봉 창에서 앞으로 10봉(2.5시간) 안에 k·σ(k=2) 를 넘는 움직임이 나오는가를 3클래스로 맞히고, 시험 구간에서 점수 상위 5% 창의 사건율을 기저 사건율로 나눈 lift 로 판정했다. 여기서는 캔들 이미지가 완전히 무력하지는 않았다({T:event}, {F:event}).")
para("5채널 캔들 이미지(차트·거래 밀도·30일 거래량·시각·요일 평면)는 흑백보다 전수 740종목 4폴드 중 3폴드에서 나았다(E1 채택, +0.25·+0.07·+0.11, 폴드 0 만 −0.07). 이미지 단독은 전체 피처 LightGBM 에 폴드마다 0.2~0.5 못 미쳤고(E2 기각) 이미지와 피처를 융합하면 흑백보다 0.2~0.4 오르지만 LightGBM 에는 0.08~0.28 못 미쳤다. LightGBM 에 이미지 점수를 스태킹한 순기여는 2025년 두 폴드에서만 유의했다(ΔAP +0.0007~+0.0031, 상대 +4%). 2024년 폴드에서는 0 이거나 음수였다(E3 체제 의존).")
table("급등·급락 발생 예측 (15m·120봉·10봉 지평, 740종목 전수, 학습 500만 창): 시험 구간 lift / 미학습 종목 lift@@event", ["모델", "폴드 0", "폴드 1", "폴드 2", "폴드 3"], [
 ["흑백 캔들 CNN", "1.94 / 1.96", "1.90 / 1.96", "1.67 / 1.67", "1.62 / 1.63"],
 ["5채널 캔들 CNN", "1.87 / 1.87", ("2.15 / 2.21",), ("1.74 / 1.73",), ("1.73 / 1.81",)],
 ["융합 f1 (이미지 + 피처 27)", "2.22 / 2.22", "2.18 / 2.22", "2.04 / 2.00", "1.83 / 1.86"],
 [("LightGBM (피처 27)",), ("2.30 / 2.33",), ("2.46 / 2.51",), ("2.27 / 2.19",), ("1.94 / 1.97",)]], widths=[4.2, 2.6, 2.6, 2.6, 2.6], size=8.5, note="lift = 점수 상위 5% 창의 사건율 ÷ 기저 사건율. 미학습 종목은 학습에 쓰지 않은 종목 홀드아웃.")
figure("fig10_event.png", "급등·급락 발생 예측의 폴드별 lift@@event", 14.5)
para("발생 예측의 가장 큰 신호는 시각·요일과 직전 거래량·변동성 비율이었다. BTC 매크로 피처와 평면(BTC 24시간 수익·시장폭)은 발생·방향 두 과제 모두에서 해로웠다. 이미지 평면으로 더하면 폴드 0·1 에서 −0.17·−0.11, LightGBM 피처로 넣어도 −0.1~−0.3 이었다. 시장 공통 상태는 표본을 시기별로 묶어 과적합을 부추긴다. 방향에서 캔들이 실패한 이유(맥락 부재)는 발생에서도 같은 방식으로 나타난다. 맥락 평면을 더한 5채널이 흑백을 이겼고 이미지 단독은 피처를 못 넘었다. 종목군을 넓히면 CNN 은 나빠지고(상위 200 흑백 2.12·1.95 → 전수 1.94·1.90) LightGBM 은 좋아졌다. 저유동 종목의 차트는 형태 정보가 적고 잡음이 커서 이미지 학습을 흐린다.")
para("이 발생 모델의 급등·급락 확률은 본문 제5장 제5절에서 방향 모델의 다리 안 비중으로 쓰였다(가설 3).")

chapter("부록 E 보유 기간과 국면별 성과")
para("보유 기간이 성과를 어떻게 바꾸는지는 같은 점수로 보유 봉수만 바꿔 재판정했다({F:decay}). 비용 후 Sharpe 는 2일 1.02, 4일 2.04, 7일 3.05, 14일 3.53, 21일 3.36, 28일 3.08, 42일 3.12 이고 비용 0 으로도 2일 3.27, 7일 3.80, 14일 3.95, 42일 3.28 로 14일이 정점이다. 짧은 보유의 약세는 왕복 0.2% 만이 아니라 신호 자체가 2주에 걸쳐 실현되기 때문이고 14일 뒤로는 6주까지 완만하게 식는다. 총수익은 어느 지평이든 ×1.55~1.72 로 비슷해 잦은 회전은 비용만 남긴다.")
figure("fig13_decay.png", "보유 기간별 4폴드 합산 Sharpe: 시드 10 앙상블 점수 고정, 보유 봉수만 변경 (비용 전·후)@@decay", 14.0)
para("시장 국면에 따라 성과가 어떻게 갈리는지는 4폴드 합산 일수익을 BTC 직전 30일 수익과 30일 실현변동성으로 나눠 보았다({T:regime}). BTC 가 30일 동안 10% 넘게 내린 117일에서 Sharpe 3.91, ±10% 안에서 움직인 251일에서 2.69, 10% 넘게 오른 40일에서 7.10 이다. 변동성이 중앙값보다 높은 날은 4.96, 낮은 날은 2.40 이다. 달러중립 롱숏이라 어느 국면에서도 양수이고 시장이 크게 움직일수록 종목 간 격차가 벌어져 성과가 좋다. 상승 국면의 7.10 은 40일 표본이라 폭보다 부호를 읽는다.")
reg = pd.read_csv("/home/arcosium/projects/HYFE_QTPA/results/ext_regime.csv")
rrows3 = [[r.regime, f"{int(r.days)}", f"{r.sharpe:.2f}", fmt(r.p, 4), f"{r.mdd:.1f}%", f"×{r.final:.2f}"] for _, r in reg.iterrows()]
table("시장 국면별 코호트 롱숏 성과 (시드 10 앙상블, 4폴드 합산 일수익을 국면으로 나눔)@@regime", ["국면 (BTC 직전 30일 기준)", "일수", "Sharpe", "NW p", "MDD", "순자산"], rrows3, widths=[4.6, 1.4, 1.8, 1.8, 1.8, 1.8], size=8.5, note="수익 국면은 BTC 직전 30일 수익 −10% 미만 / ±10% / +10% 초과, 변동성 국면은 30일 실현변동성의 중앙값 기준. 각 국면의 일수익을 이어 붙여 Sharpe 를 계산.")

tok = re.compile(r"\{([TF]):([a-z0-9_]+)\}")
def _sub(t): return tok.sub(lambda m: (f"<표 {table.keys[m.group(2)]}>" if m.group(1) == "T" else f"<그림 {figure.keys[m.group(2)]}>"), t)
for _p in doc.paragraphs:
    for _r in _p.runs:
        if tok.search(_r.text): _r.text = _sub(_r.text)   # 그림 run 은 건드리지 않는다(run.text 대입이 drawing 을 지운다)
for _t in doc.tables:
    for _row in _t.rows:
        for _c in _row.cells:
            for _p in _c.paragraphs:
                for _r in _p.runs:
                    if tok.search(_r.text): _r.text = _sub(_r.text)
add_page_number_footer()
doc.save(OUT); print("saved", OUT)
