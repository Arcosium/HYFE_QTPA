"""Read-only Binance funding history; coverage remains explicit.

Binance rates are the paper's proxy for other venues, not their actual charges.
API history is recursively partitioned so a full 1000-row page cannot silently
truncate simultaneous settlements. The private cache is safe to rebuild.
"""
import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path

from hyfe.paper_models import atomic_json

API = "https://fapi.binance.com/fapi/v1/"


def get_json(endpoint, params=None):
    url = API + endpoint + ("?" + urllib.parse.urlencode(params) if params else "")
    request = urllib.request.Request(url, headers={"User-Agent": "AutoCrypto-paper/1"})
    with urllib.request.urlopen(request, timeout=12) as response:
        return json.load(response)


def history(start, end, symbols, request=get_json):
    if end < start:
        return []
    rows = request("fundingRate", {"startTime": int(start), "endTime": int(end), "limit": 1000})
    if not isinstance(rows, list):
        raise ValueError("invalid funding response")
    if len(rows) < 1000:
        return rows
    if end > start:
        mid = (start + end) // 2
        return history(start, mid, symbols, request) + history(mid + 1, end, symbols, request)
    out = []
    for symbol in symbols:
        part = request("fundingRate", {"symbol": symbol, "startTime": start, "endTime": end, "limit": 1000})
        if not isinstance(part, list) or len(part) >= 1000:
            raise ValueError("funding page could not be completed")
        out.extend(part)
    return out


def collect(runtime, boundaries, bases, request=get_json):
    """Return rates over each (previous 4h close, close], including known zeros."""
    if not boundaries:
        return {}
    from hyfe.paper_rules import STEP_MS
    runtime = Path(runtime)
    infofile = runtime / "funding_symbols.json"
    if infofile.exists() and time.time() - infofile.stat().st_mtime < 86400:
        instruments = json.loads(infofile.read_text())
    else:
        data = request("exchangeInfo")
        instruments = {s["baseAsset"]: s["symbol"] for s in data["symbols"]
                       if s.get("quoteAsset") == "USDT" and s.get("contractType") == "PERPETUAL"}
        atomic_json(infofile, instruments)
    aliases = {b: instruments[b] for b in bases if b in instruments}
    start, end = min(boundaries) - STEP_MS + 1, max(boundaries)
    cache = runtime / "funding" / f"{start}-{end}.json"
    if cache.exists():
        rows = json.loads(cache.read_text())
    else:
        rows = history(start, end, list(aliases.values()), request)
        for r in rows:
            if not math.isfinite(float(r["fundingRate"])):
                raise ValueError("nonfinite funding rate")
        atomic_json(cache, rows)
    by_symbol = {}
    for r in rows:
        by_symbol.setdefault(r["symbol"], {})[int(r["fundingTime"])] = float(r["fundingRate"])
    out = {}
    for close in boundaries:
        out[close] = {b: sum(rate for ts, rate in by_symbol.get(symbol, {}).items()
                             if close - STEP_MS < ts <= close) for b, symbol in aliases.items()}
    return out
