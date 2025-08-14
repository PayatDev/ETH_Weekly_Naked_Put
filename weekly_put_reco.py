# minimal_weekly_put_reco.py
import os, json, math, re, requests, numpy as np
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

TH = ZoneInfo("Asia/Bangkok")
MONTHS = {m:i+1 for i,m in enumerate(["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"])}

# ---------- utils ----------
def _phi(x): return 0.5*(1+math.erf(x/math.sqrt(2)))
def _round_to(x, step): return round(x/step)*step

def bs_put_prob(S, K, T, vol):
    if not(S>0 and K>0 and T>0 and vol and vol>0): return None, None
    sT = math.sqrt(T); d1 = (math.log(S/K)+(0.5*vol*vol)*T)/(vol*sT); d2 = d1 - vol*sT
    return float(_phi(d2)), float(_phi(d1)-1.0)  # prob_success, delta_put

def parse_deribit_expiry(sym):
    m = re.match(r"^[A-Z]+-(\d{2})([A-Z]{3})(\d{2})", sym)
    if not m: return None
    day, mon, yy = int(m[1]), m[2], 2000+int(m[3])
    return datetime(yy, MONTHS.get(mon,1), day, 8, 0, tzinfo=timezone.utc)

# ---------- Function 1: fetch LLM input ----------
def get_eth_data(ohlc_days=90):
    """ดึง ETH OHLCV รายวันจาก Drift + TA + Fear&Greed + Deribit weekly PUT → JSON สำหรับ LLM"""
    # 1) Drift OHLC daily
    recs = requests.get(
        "https://data.api.drift.trade/market/ETH-PERP/candles/D",
        params={"limit": str(max(ohlc_days, 30))}, timeout=10
    ).json().get("records", [])
    if not recs: raise RuntimeError("No daily candles from Drift")
    recs.sort(key=lambda r: r["ts"])  # เก่า→ใหม่

    ohlc_hist, closes = [], []
    for r in recs[-ohlc_days:]:
        ts = int(r["ts"])
        dt_th = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(TH)
        ohlc_hist.append({
            "date": dt_th.strftime("%Y-%m-%d"),
            "open": float(r["fillOpen"]),
            "high": float(r["fillHigh"]),
            "low":  float(r["fillLow"]),
            "close":float(r["fillClose"]),
            "volume_base":  float(r.get("baseVolume", 0.0)),
            "volume_quote": float(r.get("quoteVolume", 0.0)),
        })
        closes.append(float(r["fillClose"]))
    spot = closes[-1]

    # 2) Fear & Greed (soft-fail)
    try:
        fg = requests.get("https://api.alternative.me/fng/", timeout=8).json()['data'][0]
        fg_val, fg_txt = int(fg['value']), fg['value_classification']
    except Exception:
        fg_val, fg_txt = None, None

    # 3) TA (มินิมอล)
    changes = [closes[i]-closes[i-1] for i in range(1, min(15, len(closes)))]
    gains = [max(0,c) for c in changes]; losses = [abs(min(0,c)) for c in changes]
    avg_g = sum(gains)/len(gains) if gains else 0.0
    avg_l = sum(losses)/len(losses) if losses else 1e-6
    rsi14 = round(100 - 100/(1 + (avg_g/avg_l)), 1)
    sma10 = round(sum(closes[-10:])/10, 2) if len(closes)>=10 else None
    sma20 = round(sum(closes[-20:])/20, 2) if len(closes)>=20 else None
    rets = [math.log(closes[i]/closes[i-1]) for i in range(1, len(closes))]
    rr = rets[-10:] if len(rets)>=10 else rets
    mu = sum(rr)/len(rr) if rr else 0.0
    var = sum((r-mu)**2 for r in rr)/len(rr) if rr else 0.0
    vol10 = round(math.sqrt(var)*math.sqrt(365)*100, 1)
    iv_est = round(vol10*1.25, 1) if vol10 is not None else None

    # 4) Deribit weekly PUT (5–9 วัน)
    arr = requests.get(
        "https://www.deribit.com/api/v2/public/get_book_summary_by_currency",
        params={"currency":"ETH","kind":"option"}, timeout=10
    ).json().get("result", []) or []
    now = datetime.now(timezone.utc)
    opts_out = []
    for o in arr:
        name = o.get("instrument_name","")
        if not name.endswith("-P"): continue
        exp = parse_deribit_expiry(name)
        if not exp: continue
        Tdays = (exp - now).total_seconds()/86400.0
        if 5 <= Tdays <= 9:
            try:
                strike = float(o.get("strike_price") or o.get("strike"))
            except:
                m = re.search(r"-(\d+(\.\d+)?)-P$", name)
                if not m: continue
                strike = float(m.group(1))
            iv = float(o["mark_iv"])/100.0 if o.get("mark_iv") else None
            prob, delta = bs_put_prob(spot, strike, max(Tdays/365.0,1e-6), iv)
            prices = [p for p in (o.get("bid_price"), o.get("ask_price"), o.get("mark_price")) if isinstance(p,(int,float))]
            mid = float(np.median(prices)) if prices else None
            opts_out.append({"strike": strike, "iv": iv, "mid_price": mid,
                             "delta_put": delta, "prob_success_bs": prob})

    return {
        "current_price": round(spot, 2),
        "rsi_14": rsi14, "sma_10": sma10, "sma_20": sma20,
        "volatility_10d": vol10, "iv_estimate": iv_est,
        "fear_greed": fg_val, "fear_greed_text": fg_txt,
        "timestamp": datetime.now(TH).strftime("%Y-%m-%d %H:%M:%S"),
        "ohlc_history": ohlc_hist,
        "options_weekly": opts_out
    }

# ---------- Function 2: send to LLM ----------
def _pick_option(options, target_delta=-0.25, min_prob=0.70):
    cand = [o for o in options if o.get("prob_success_bs") is not None and o.get("mid_price") is not None]
    if not cand: return None
    cand.sort(key=lambda o: (abs((o.get("delta_put") or target_delta)-target_delta), -(o.get("prob_success_bs") or 0)))
    for o in cand:
        if o["prob_success_bs"] >= min_prob: return o
    return cand[0]

def build_llm_payload(d, max_days=90, step=1):
    spot = float(d["current_price"])
    support_hint = d.get("sma_20") or _round_to(spot*0.95, 50)
    resistance_hint = _round_to(spot*1.03, 50)
    hist = d.get("ohlc_history", [])[-max_days:]
    if step>1: hist = [hist[i] for i in range(0, len(hist), step)]
    return {
        "now_th": datetime.now(TH).strftime("%Y-%m-%d %H:%M"),
        "spot": spot,
        "ta": {"rsi_14": d.get("rsi_14"), "sma_10": d.get("sma_10"),
               "sma_20": d.get("sma_20"), "volatility_10d_pct": d.get("volatility_10d"),
               "iv_estimate_pct": d.get("iv_estimate")},
        "sentiment": {"fear_greed": d.get("fear_greed"), "fear_greed_text": d.get("fear_greed_text")},
        "support_hint": support_hint, "resistance_hint": resistance_hint,
        "ohlc_history": hist,
        "candidate_option": _pick_option(d.get("options_weekly", []))
    }

def run_llm_recommendation(llm_input, model="gpt-4o-mini", max_days=90, step=1):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key: raise RuntimeError("Please set OPENAI_API_KEY")
    payload = build_llm_payload(llm_input, max_days=max_days, step=step)

    system = ("คุณเป็นผู้ช่วยกลยุทธ์ขาย ETH weekly naked put เน้นวินัยความเสี่ยง "
              "ตอบเฉพาะ JSON เท่านั้น")
    user = (
        "สรุปตามสเปก:\n"
        "- ตอนนี้ (เวลาไทย) เหมาะ/ไม่เหมาะ + เหตุผลสั้นๆ 1 ประโยค\n"
        "- แนวโน้ม 5–7 วัน: ขึ้น/ลง/ไซด์เวย์ + support/resistance อย่างละ 1 ตัว (ใช้ hint ได้)\n"
        "- เลือก strike PUT จาก candidate (ถ้าไม่เหมาะ ปรับ strike ใกล้เดลต้า -0.25 และ prob สูง)\n"
        "- ตอบ JSON ฟิลด์: suitability_now, reason, path_5_7d, support, resistance, strike_put, prob_success_bs, action\n\n"
        f"DATA:\n{json.dumps(payload, ensure_ascii=False)}\n"
        "ตอบเป็น JSON เพียวๆ"
    )
    body = {
        "model": model,
        "messages": [{"role":"system","content":system},{"role":"user","content":user}],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        data=json.dumps(body), timeout=30
    )
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except Exception:
        import re
        m = re.search(r"\{.*\}", content, re.S)
        return json.loads(m.group(0)) if m else {"error":"LLM did not return JSON"}

# ---------- Function 3: format & Telegram ----------
def format_report_th(llm_input: dict, llm_reco: dict) -> str:
    spot = llm_input.get("current_price")
    rsi  = llm_input.get("rsi_14")
    sma20 = llm_input.get("sma_20")
    vol10 = llm_input.get("volatility_10d")
    iv   = llm_input.get("iv_estimate")
    fg_v = llm_input.get("fear_greed")
    fg_t = llm_input.get("fear_greed_text")
    ts   = llm_input.get("timestamp")

    suit = llm_reco.get("suitability_now")
    reason = llm_reco.get("reason")
    path = llm_reco.get("path_5_7d")
    sup  = llm_reco.get("support")
    res  = llm_reco.get("resistance")
    strike = llm_reco.get("strike_put")
    prob  = llm_reco.get("prob_success_bs")
    action = llm_reco.get("action")

    prob_txt = f"{prob:.2%}" if isinstance(prob, (int,float)) else "N/A"

    msg = (
        f"🧠 ETH Weekly Naked Put — {ts}\n"
        f"• Spot: ${spot:,.2f} | RSI14: {rsi} | SMA20: {sma20}\n"
        f"• Vol(10d): {vol10}% | IV est: {iv}%\n"
        f"• Fear&Greed: {fg_v} ({fg_t})\n"
        f"\n"
        f"📈 แนวโน้ม 5–7 วัน: {path}\n"
        f"• Support: {sup} | Resistance: {res}\n"
        f"\n"
        f"📝 คำแนะนำตอนนี้: {suit} — {reason}\n"
        f"• PUT strike: {strike} | Prob: {prob_txt}\n"
        f"• Action: {str(action).upper()}\n"
    )
    return msg

def send_telegram_message(bot_token: str, chat_id: str, text: str) -> bool:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"❌ Telegram send failed: {e}")
        try:
            print("Response:", r.text)  # ถ้ามี response กลับมา
        except:
            pass
        return False

# ---------- quick test ----------
if __name__ == "__main__":

    import os
    os.environ["OPENAI_API_KEY"] = "sk-proj-Oag4vOJH8YUNUJxbTlqbEj5.....25aAzLoUTawcxa9P2VE_8D6mRPVwp7Qc2z5mjg3OcbyXBYOY6JIA"

    # ---- CONFIG ----
    BOT_TOKEN = "8300561341:AA......OXPXCsUjmdo04UXVPU"
    CHAT_ID = "166..48244"

    data = get_eth_data(ohlc_days=90)
    print("INPUT PREVIEW:", json.dumps(
        {k:data[k] for k in ["current_price","rsi_14","sma_20","volatility_10d","fear_greed"]},
        ensure_ascii=False, indent=2
    ))
    reco = run_llm_recommendation(data, max_days=90, step=1)  # step=2/3 เพื่อลด token ได้
    print("LLM RECO:", json.dumps(reco, ensure_ascii=False, indent=2))

    # 3) ส่ง Telegram
    report = format_report_th(data, reco)
    sent = send_telegram_message(BOT_TOKEN, CHAT_ID, report)
    print("Telegram:", "✅ sent" if sent else "❌ failed")
