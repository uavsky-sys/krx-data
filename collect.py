# -*- coding: utf-8 -*-
"""
KRX 데이터 수집기 v2 (GitHub Actions에서 매일 자동 실행)

v1 대비 바뀐 점 — watchlist에 없는 종목도 미리 다 받아둔다
- 코스피+코스닥 '시가총액 상위 N종목'을 자동으로 잡아 지표를 계산한다 (기본 900종목)
- 결과를 data/indicators/all.json 한 파일에 모아 담는다 → 어떤 종목이든 즉시 조회 가능
- 시장 전체 통계(정배열 비율·MA200 위 비율·평균 RSI)와 스크리닝 결과를 screen.md로 만든다
- watchlist.txt 종목은 그대로 '고정 추적'으로 남아 history CSV·개별 JSON·summary.md를 유지한다

초보자 주의: 이 파일은 고칠 필요가 없습니다.
  - 수집 종목 수를 바꾸려면 GitHub 저장소 Settings > Secrets and variables > Actions >
    Variables 탭에서 TOP_N 을 만들어 숫자를 넣으면 됩니다. (안 만들면 900)
  - 시총 900위 밖 종목을 꼭 추적하려면 watchlist.txt 에 추가하세요.
"""
import os, sys, json, time, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
HIST = os.path.join(DATA, "history")
IND = os.path.join(DATA, "indicators")
DAILY = os.path.join(DATA, "daily")
for d in (DATA, HIST, IND, DAILY):
    os.makedirs(d, exist_ok=True)

KST = datetime.timezone(datetime.timedelta(hours=9))
TODAY = datetime.datetime.now(KST).date()
NOW_KST = datetime.datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")

# 수집 규모 / 동시 실행 수 (환경변수로 조절 가능)
TOP_N = int(os.environ.get("TOP_N", "900") or "900")
WORKERS = int(os.environ.get("WORKERS", "6") or "6")


def read_watchlist():
    """watchlist.txt에서 종목코드 목록을 읽는다. 형식: '005930 삼성전자' (한 줄에 하나)"""
    items = []
    path = os.path.join(BASE, "watchlist.txt")
    if not os.path.exists(path):
        return items
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            code = parts[0]
            name = parts[1] if len(parts) > 1 else code
            if code.isdigit() and len(code) == 6:
                items.append((code, name))
    return items


# ---------------------------------------------------------------- 지표 계산

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """RSI(상대강도지수) — Wilder 방식"""
    d = close.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    au = up.ewm(alpha=1 / n, min_periods=n).mean()
    ad = dn.ewm(alpha=1 / n, min_periods=n).mean()
    rs = au / ad
    return 100 - 100 / (1 + rs)


def fetch_history(code: str, days: int = 740, retry: int = 2) -> pd.DataFrame:
    """FinanceDataReader로 최근 2년 일봉을 받는다 (로그인 불필요). 실패 시 재시도."""
    import FinanceDataReader as fdr

    start = (TODAY - datetime.timedelta(days=days)).isoformat()
    last = None
    for i in range(retry + 1):
        try:
            df = fdr.DataReader(code, start)
            if df is None or len(df) == 0:
                raise ValueError("빈 데이터")
            df = df.rename(columns=str.title)  # Open High Low Close Volume
            df.index.name = "Date"
            return df
        except Exception as e:          # 네트워크 순간 오류 등
            last = e
            time.sleep(1.5 * (i + 1))
    raise last


def compute_indicators(code: str, name: str, df: pd.DataFrame) -> dict:
    c = df["Close"]
    v = df["Volume"] if "Volume" in df.columns else pd.Series([0] * len(c), index=c.index)
    value = c * v  # 근사 거래대금
    out = {
        "종목코드": code,
        "종목명": name,
        "기준일": str(df.index[-1].date()),
        "종가": float(c.iloc[-1]),
        "등락률(%)": round(float(c.pct_change().iloc[-1] * 100), 2),
    }
    for n in (5, 20, 60, 120, 200):
        if len(c) >= n:
            ma = float(c.rolling(n).mean().iloc[-1])
            out[f"MA{n}"] = round(ma, 1)
            out[f"MA{n}_이격(%)"] = round((float(c.iloc[-1]) / ma - 1) * 100, 2)
        else:
            out[f"MA{n}"] = None
    r = rsi(c)
    out["RSI14"] = round(float(r.iloc[-1]), 1) if len(c) >= 15 else None
    out["RSI14_전일"] = round(float(r.iloc[-2]), 1) if len(c) >= 16 else None
    # 거래대금 신호: 오늘 거래대금이 최근 20일 평균의 몇 배인가
    if len(value) >= 21:
        avg20 = float(value.rolling(20).mean().iloc[-2])
        out["거래대금_20일평균대비(배)"] = round(float(value.iloc[-1]) / avg20, 2) if avg20 else None
    hi52 = float(c[-250:].max()) if len(c) >= 2 else float(c.max())
    lo52 = float(c[-250:].min()) if len(c) >= 2 else float(c.min())
    out["52주고가"] = hi52
    out["52주저가"] = lo52
    out["52주고가대비(%)"] = round((float(c.iloc[-1]) / hi52 - 1) * 100, 1)
    # 20거래일 누적 등락률 (지침 §6 고변동성 판정 입력값)
    if len(c) >= 21:
        out["20거래일누적(%)"] = round((float(c.iloc[-1]) / float(c.iloc[-21]) - 1) * 100, 1)
    # 정배열/역배열 판정
    mas = [out.get(f"MA{n}") for n in (20, 60, 120)]
    if all(m is not None for m in mas):
        if mas[0] > mas[1] > mas[2]:
            out["배열"] = "정배열(20>60>120)"
        elif mas[0] < mas[1] < mas[2]:
            out["배열"] = "역배열(20<60<120)"
        else:
            out["배열"] = "혼조"
    return out


# ---------------------------------------------------------------- 유니버스

def _pick(df: pd.DataFrame, *names):
    """여러 후보 컬럼명 중 실제로 존재하는 첫 번째를 돌려준다 (fdr 버전차 대응)"""
    for n in names:
        if n in df.columns:
            return n
    return None


def build_universe(top_n: int):
    """코스피+코스닥 시가총액 상위 top_n 종목을 (코드, 이름, 시총, 시장) 리스트로 만든다.

    FinanceDataReader 버전에 따라 컬럼명이 달라서 방어적으로 처리한다.
    실패하면 빈 리스트를 돌려주고, 호출부는 watchlist만으로 계속 진행한다.
    """
    import FinanceDataReader as fdr

    frames = []
    try:
        df = fdr.StockListing("KRX")
        if df is not None and len(df):
            frames.append(df)
    except Exception as e:
        print(f"  StockListing('KRX') 실패: {e}")
    if not frames:
        for mkt in ("KOSPI", "KOSDAQ"):
            try:
                d = fdr.StockListing(mkt)
                if d is not None and len(d):
                    if "Market" not in d.columns:
                        d = d.assign(Market=mkt)
                    frames.append(d)
            except Exception as e:
                print(f"  StockListing('{mkt}') 실패: {e}")
    if not frames:
        print("  ⚠ 종목 목록 확보 실패 → watchlist 종목만 수집합니다")
        return []

    df = pd.concat(frames, ignore_index=True)

    c_code = _pick(df, "Code", "Symbol", "종목코드")
    c_name = _pick(df, "Name", "종목명")
    c_cap = _pick(df, "Marcap", "MarketCap", "시가총액")
    c_mkt = _pick(df, "Market", "MarketId", "시장구분")
    c_close = _pick(df, "Close", "종가")
    c_shares = _pick(df, "Stocks", "Shares", "상장주식수")

    if c_code is None or c_name is None:
        print(f"  ⚠ 컬럼 인식 실패 (columns={list(df.columns)[:15]}) → watchlist만 수집")
        return []

    ren = {}
    for src, dst in ((c_code, "code"), (c_name, "name"), (c_cap, "cap"),
                     (c_mkt, "mkt"), (c_close, "close"), (c_shares, "shares")):
        if src and src not in ren:
            ren[src] = dst
    df = df[list(ren.keys())].rename(columns=ren).copy()

    # 시가총액 확보: 없으면 종가 × 상장주식수로 계산
    if "cap" not in df.columns:
        if "close" in df.columns and "shares" in df.columns:
            df["cap"] = pd.to_numeric(df["close"], errors="coerce") * \
                        pd.to_numeric(df["shares"], errors="coerce")
            print("  시가총액 컬럼 없음 → 종가×상장주식수로 계산")
        else:
            print("  ⚠ 시가총액 확보 실패 → 목록 순서 상위로 대체")
            df["cap"] = 0
    df["cap"] = pd.to_numeric(df["cap"], errors="coerce").fillna(0)

    df["code"] = df["code"].astype(str).str.strip().str.zfill(6)
    df["name"] = df["name"].astype(str).str.strip()
    if "mkt" not in df.columns:
        df["mkt"] = ""
    df["mkt"] = df["mkt"].astype(str)

    # 6자리 숫자 코드만 / 스팩·우선주 제외 / 중복 제거
    df = df[df["code"].str.fullmatch(r"\d{6}")]
    # 스팩 제외 / 우선주 제외(이름이 '우'·'우B'로 끝나면서 코드 끝자리가 0이 아닌 것)
    is_spac = df["name"].str.contains("스팩", na=False)
    is_pref = df["name"].str.contains(r"우B?$", regex=True, na=False) & \
              (~df["code"].str.endswith("0"))
    df = df[~(is_spac | is_pref)]
    df = df.drop_duplicates(subset=["code"]).sort_values("cap", ascending=False)

    uni = [(r.code, r.name, float(r.cap), r.mkt) for r in df.head(top_n).itertuples()]
    if uni:
        print(f"  유니버스 {len(uni)}종목 (시총 1위 {uni[0][1]} / 최하위 {uni[-1][1]} "
              f"{uni[-1][2]/1e8:,.0f}억)")
    return uni


def _one(code, name, cap, mkt, save_history=False):
    df = fetch_history(code)
    if save_history:
        df.to_csv(os.path.join(HIST, f"{code}.csv"), encoding="utf-8-sig")
    ind = compute_indicators(code, name, df)
    if cap:
        ind["시가총액(억)"] = round(cap / 1e8, 1)
    if mkt:
        ind["시장"] = mkt
    return ind


# ---------------------------------------------------------------- 출력물

def write_summary(rows):
    """watchlist(고정 추적) 종목 요약표 — v1과 동일 포맷 유지"""
    if not rows:
        return
    lines = [f"# 기술지표 요약 — {rows[0]['기준일']} 종가 기준", "",
             "| 종목 | 종가 | 등락률 | MA20 | MA60 | MA120 | RSI14 | 배열 | 거래대금(20일比) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in rows:
        lines.append(
            f"| {s['종목명']}({s['종목코드']}) | {s['종가']:,.0f} | {s['등락률(%)']}% "
            f"| {s.get('MA20') or '—'} | {s.get('MA60') or '—'} | {s.get('MA120') or '—'} "
            f"| {s.get('RSI14') or '—'} | {s.get('배열','—')} | {s.get('거래대금_20일평균대비(배)','—')} |")
    with open(os.path.join(IND, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _tbl(title, rows, note="", limit=20):
    """스크리닝 표 한 덩어리. 잘린 개수를 반드시 표기한다(조용한 누락 금지)."""
    out = [f"## {title}  ({len(rows)}종목)"]
    if note:
        out.append(f"> {note}")
    out.append("")
    if not rows:
        out += ["해당 없음", ""]
        return out
    out += ["| 종목 | 코드 | 종가 | 등락률 | RSI14 | MA20이격 | MA200이격 | 거래대금(20일比) | 시총(억) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for s in rows[:limit]:
        out.append(
            f"| {s['종목명']} | {s['종목코드']} | {s['종가']:,.0f} | {s['등락률(%)']}% "
            f"| {s.get('RSI14','—')} | {s.get('MA20_이격(%)','—')} | {s.get('MA200_이격(%)','—')} "
            f"| {s.get('거래대금_20일평균대비(배)','—')} | {s.get('시가총액(억)','—')} |")
    if len(rows) > limit:
        out.append(f"\n※ 상위 {limit}종목만 표기. **{len(rows) - limit}종목 생략됨.** "
                   f"전체는 `all.json`에서 확인.")
    out.append("")
    return out


def write_screen(rows):
    """전 종목 시장 통계 + 스크리닝"""
    if not rows:
        return
    base = rows[0]["기준일"]
    n = len(rows)

    def g(s, k):
        v = s.get(k)
        return v if isinstance(v, (int, float)) else None

    ma200 = [s for s in rows if g(s, "MA200_이격(%)") is not None]
    above200 = [s for s in ma200 if s["MA200_이격(%)"] > 0]
    jung = [s for s in rows if s.get("배열", "").startswith("정배열")]
    yeok = [s for s in rows if s.get("배열", "").startswith("역배열")]
    rsis = [g(s, "RSI14") for s in rows if g(s, "RSI14") is not None]
    ups = [s for s in rows if g(s, "등락률(%)") is not None and s["등락률(%)"] > 0]

    L = [f"# 시장 전체 스크리닝 — {base} 종가 기준", "",
         f"> 대상: 코스피+코스닥 시가총액 상위 **{n}종목** / 생성 {NOW_KST}", "",
         "## 시장 폭(Breadth) — 국면 판정용", "",
         "| 지표 | 값 |", "|---|---|",
         f"| 상승 종목 비율 | {len(ups)/n*100:.1f}% ({len(ups)}/{n}) |",
         f"| MA200 **위** 종목 비율 | {len(above200)/len(ma200)*100:.1f}% ({len(above200)}/{len(ma200)}) |"
         if ma200 else "| MA200 위 비율 | 확인 불가 |",
         f"| 정배열 종목 | {len(jung)/n*100:.1f}% ({len(jung)}) |",
         f"| 역배열 종목 | {len(yeok)/n*100:.1f}% ({len(yeok)}) |",
         f"| 평균 RSI14 | {sum(rsis)/len(rsis):.1f} |" if rsis else "| 평균 RSI14 | 확인 불가 |",
         f"| RSI 30 이하(과매도) | {len([x for x in rsis if x <= 30])}종목 |" if rsis else "",
         f"| RSI 70 이상(과매수) | {len([x for x in rsis if x >= 70])}종목 |" if rsis else "",
         "",
         "> 해석 기준: MA200 위 비율이 30% 아래면 약세장, 70% 위면 강세장으로 통용됩니다. "
         "정배열/역배열 비율과 함께 보십시오.", ""]

    over = sorted([s for s in rows if g(s, "RSI14") is not None and s["RSI14"] <= 30],
                  key=lambda s: s["RSI14"])
    hot = sorted([s for s in rows if g(s, "RSI14") is not None and s["RSI14"] >= 70],
                 key=lambda s: -s["RSI14"])
    vol = sorted([s for s in rows if g(s, "거래대금_20일평균대비(배)") is not None
                  and s["거래대금_20일평균대비(배)"] >= 3],
                 key=lambda s: -s["거래대금_20일평균대비(배)"])
    fall = sorted([s for s in rows if g(s, "52주고가대비(%)") is not None
                   and s["52주고가대비(%)"] <= -50],
                  key=lambda s: s["52주고가대비(%)"])
    trend = sorted([s for s in jung if g(s, "MA200_이격(%)") is not None
                    and s["MA200_이격(%)"] > 0], key=lambda s: -(s.get("시가총액(억)") or 0))

    L += _tbl("과매도 — RSI14 30 이하", over, "낙폭이 컸다는 사실일 뿐, 반등 신호가 아닙니다.")
    L += _tbl("과매수 — RSI14 70 이상", hot, "단기 과열. 추격 진입 위험 구간.")
    L += _tbl("거래대금 급증 — 20일 평균의 3배 이상", vol, "재료 발생 가능성. 방향은 별도 확인 필요.")
    L += _tbl("낙폭과대 — 52주 고점 대비 −50% 이하", fall, "싸다는 뜻이 아니라 많이 빠졌다는 뜻입니다.")
    L += _tbl("추세 양호 — 정배열 + MA200 위 (시총순)", trend)

    L += ["---", "",
          "*정보 제공물이며 투자 권유가 아닙니다. 기술적 지표는 국면 판단의 한 축일 뿐입니다.*"]
    with open(os.path.join(IND, "screen.md"), "w", encoding="utf-8") as f:
        f.write("\n".join([x for x in L if x is not None]) + "\n")


# ---------------------------------------------------------------- KRX(선택)

def krx_snapshot():
    """(선택) KRX OpenAPI 키가 있으면 전 종목 일별 시세 스냅샷 저장."""
    key = os.environ.get("KRX_API_KEY", "").strip()
    if not key:
        print("KRX_API_KEY 없음 → 전 종목 스냅샷 생략 (지표 수집은 정상 진행)")
        return
    import requests

    bas_dd = None
    for back in range(0, 7):
        d = (TODAY - datetime.timedelta(days=back)).strftime("%Y%m%d")
        ok = False
        for mkt, ep in [("kospi", "stk_bydd_trd"), ("kosdaq", "ksq_bydd_trd")]:
            for scheme in ("https", "http"):
                url = f"{scheme}://data-dbg.krx.co.kr/svc/apis/sto/{ep}"
                try:
                    r = requests.get(url, headers={"AUTH_KEY": key},
                                     params={"basDd": d}, timeout=20)
                    if r.status_code == 200 and r.json().get("OutBlock_1"):
                        rows = r.json()["OutBlock_1"]
                        pd.DataFrame(rows).to_csv(
                            os.path.join(DAILY, f"{d}_{mkt}.csv"),
                            index=False, encoding="utf-8-sig")
                        ok = True
                        break
                except Exception:
                    continue
        if ok:
            bas_dd = d
            break
    print(f"KRX 스냅샷: {bas_dd or '실패 — 지표 수집에는 영향 없음'}")


# ---------------------------------------------------------------- 메인

def main():
    watch = read_watchlist()
    manual = []
    for arg in sys.argv[1:]:
        arg = arg.strip()
        if arg.isdigit() and len(arg) == 6 and arg not in [c for c, _ in watch]:
            manual.append((arg, arg))
    watch += manual
    watch_codes = {c for c, _ in watch}
    print(f"고정 추적(watchlist{'+수동' if manual else ''}): {len(watch)}종목")

    print(f"유니버스 구성 중 (시총 상위 {TOP_N})...")
    try:
        uni = build_universe(TOP_N)
    except Exception as e:
        print(f"  유니버스 구성 실패: {e} → watchlist만 수집")
        uni = []

    # 작업 목록: watchlist는 history 저장 O, 나머지는 지표만
    jobs = [(c, n, 0.0, "", True) for c, n in watch]
    seen = set(watch_codes)
    for code, name, cap, mkt in uni:
        if code in seen:
            # watchlist 종목이면 시총·시장만 나중에 병합
            continue
        jobs.append((code, name, cap, mkt, False))
        seen.add(code)
    cap_map = {code: (cap, mkt) for code, _, cap, mkt in uni}

    print(f"총 {len(jobs)}종목 수집 시작 (동시 {WORKERS})")
    results, fails = {}, []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_one, c, n, cap, mkt, sh): (c, n)
                for c, n, cap, mkt, sh in jobs}
        done = 0
        for fu in as_completed(futs):
            code, name = futs[fu]
            done += 1
            try:
                results[code] = fu.result()
            except Exception as e:
                fails.append((code, name, str(e)[:80]))
            if done % 50 == 0:
                print(f"  ...{done}/{len(jobs)} ({time.time()-t0:.0f}초, 실패 {len(fails)})")

    # watchlist 종목에도 시총·시장 붙이기
    for code, ind in results.items():
        if code in cap_map and "시가총액(억)" not in ind:
            cap, mkt = cap_map[code]
            if cap:
                ind["시가총액(억)"] = round(cap / 1e8, 1)
            if mkt:
                ind["시장"] = mkt

    # ① watchlist 개별 JSON + summary.md (v1 호환)
    wrows = []
    for code, name in watch:
        ind = results.get(code)
        if not ind:
            continue
        with open(os.path.join(IND, f"{code}.json"), "w", encoding="utf-8") as f:
            json.dump(ind, f, ensure_ascii=False, indent=2)
        wrows.append(ind)
    write_summary(wrows)

    # ② 전 종목 all.json
    allrows = sorted(results.values(), key=lambda s: -(s.get("시가총액(억)") or 0))
    base = allrows[0]["기준일"] if allrows else str(TODAY)
    payload = {
        "기준일": base,
        "생성시각": NOW_KST,
        "종목수": len(allrows),
        "수집대상": f"코스피+코스닥 시총 상위 {TOP_N} + watchlist",
        "실패": [{"종목코드": c, "종목명": n, "사유": m} for c, n, m in fails],
        "종목": allrows,
    }
    with open(os.path.join(IND, "all.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    # ③ 스크리닝
    write_screen(allrows)

    krx_snapshot()
    print(f"완료: {len(results)}/{len(jobs)} 종목, {time.time()-t0:.0f}초, 실패 {len(fails)}건")
    for c, n, m in fails[:20]:
        print(f"  FAIL {c} {n}: {m}")
    if len(fails) > 20:
        print(f"  ...외 {len(fails)-20}건 실패")


if __name__ == "__main__":
    main()
