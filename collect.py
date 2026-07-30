# -*- coding: utf-8 -*-
"""
KRX 데이터 수집기 (GitHub Actions에서 매일 자동 실행)
- watchlist.txt의 종목별로 2년치 일봉을 받아 data/history/에 저장
- 이동평균선(5·20·60·120·200일)·RSI(14)·거래대금 신호를 계산해 data/indicators/에 저장
- (선택) KRX OpenAPI 인증키가 등록돼 있으면 전 종목 일별 시세를 data/daily/에 저장
초보자 주의: 이 파일은 고칠 필요가 없습니다. 종목 추가는 watchlist.txt만 수정하면 됩니다.
"""
import os, sys, json, datetime

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


def read_watchlist():
    """watchlist.txt에서 종목코드 목록을 읽는다. 형식: '005930 삼성전자' (한 줄에 하나)"""
    items = []
    path = os.path.join(BASE, "watchlist.txt")
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


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """RSI(상대강도지수) — Wilder 방식"""
    d = close.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    au = up.ewm(alpha=1 / n, min_periods=n).mean()
    ad = dn.ewm(alpha=1 / n, min_periods=n).mean()
    rs = au / ad
    return 100 - 100 / (1 + rs)


def fetch_history(code: str) -> pd.DataFrame:
    """FinanceDataReader로 최근 2년 일봉을 받는다 (로그인 불필요)"""
    import FinanceDataReader as fdr

    start = (TODAY - datetime.timedelta(days=740)).isoformat()
    df = fdr.DataReader(code, start)
    df = df.rename(columns=str.title)  # Open High Low Close Volume
    df.index.name = "Date"
    return df


def compute_indicators(code: str, name: str, df: pd.DataFrame) -> dict:
    c = df["Close"]
    v = df["Volume"]
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


def krx_snapshot():
    """(선택) KRX OpenAPI 키가 있으면 전 종목 일별 시세 스냅샷 저장.
    키 등록: GitHub 저장소 Settings > Secrets > Actions > KRX_API_KEY"""
    key = os.environ.get("KRX_API_KEY", "").strip()
    if not key:
        print("KRX_API_KEY 없음 → 전 종목 스냅샷 생략 (watchlist 수집은 정상 진행)")
        return
    import requests

    bas_dd = None
    # 오늘부터 거슬러 최근 영업일 데이터를 찾는다 (KRX는 전 영업일까지 제공)
    for back in range(0, 7):
        d = (TODAY - datetime.timedelta(days=back)).strftime("%Y%m%d")
        ok = False
        for mkt, ep in [("kospi", "stk_bydd_trd"), ("kosdaq", "ksq_bydd_trd")]:
            for scheme in ("https", "http"):
                url = f"{scheme}://data-dbg.krx.co.kr/svc/apis/sto/{ep}"
                try:
                    r = requests.get(url, params={"basDd": d}, headers={"AUTH_KEY": key}, timeout=30)
                    rows = r.json().get("OutBlock_1", [])
                    if rows:
                        pd.DataFrame(rows).to_csv(
                            os.path.join(DAILY, f"{d[:4]}-{d[4:6]}-{d[6:]}_{mkt}.csv"),
                            index=False, encoding="utf-8-sig")
                        print(f"KRX 스냅샷 저장: {d} {mkt} {len(rows)}종목")
                        ok = True
                    break
                except Exception as e:
                    print(f"KRX {ep} {scheme} 실패: {e}")
        if ok:
            bas_dd = d
            break
    if not bas_dd:
        print("KRX 스냅샷 실패 — watchlist 수집에는 영향 없음")


def main():
    watch = read_watchlist()
    # 수동 실행 시 임시 종목코드를 인자로 받을 수 있음
    for arg in sys.argv[1:]:
        arg = arg.strip()
        if arg.isdigit() and len(arg) == 6 and arg not in [c for c, _ in watch]:
            watch.append((arg, arg))

    summary = []
    for code, name in watch:
        try:
            df = fetch_history(code)
            df.to_csv(os.path.join(HIST, f"{code}.csv"), encoding="utf-8-sig")
            ind = compute_indicators(code, name, df)
            with open(os.path.join(IND, f"{code}.json"), "w", encoding="utf-8") as f:
                json.dump(ind, f, ensure_ascii=False, indent=2)
            summary.append(ind)
            print(f"OK {code} {name}: 종가 {ind['종가']:,.0f} RSI {ind.get('RSI14')}")
        except Exception as e:
            print(f"FAIL {code} {name}: {e}")

    # 사람이 읽는 요약표
    if summary:
        lines = [f"# 기술지표 요약 — {summary[0]['기준일']} 종가 기준", "",
                 "| 종목 | 종가 | 등락률 | MA20 | MA60 | MA120 | RSI14 | 배열 | 거래대금(20일比) |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for s in summary:
            lines.append(
                f"| {s['종목명']}({s['종목코드']}) | {s['종가']:,.0f} | {s['등락률(%)']}% "
                f"| {s.get('MA20') or '—'} | {s.get('MA60') or '—'} | {s.get('MA120') or '—'} "
                f"| {s.get('RSI14') or '—'} | {s.get('배열','—')} | {s.get('거래대금_20일평균대비(배)','—')} |")
        with open(os.path.join(IND, "summary.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    krx_snapshot()
    print(f"완료: {len(summary)}/{len(watch)} 종목")


if __name__ == "__main__":
    main()
