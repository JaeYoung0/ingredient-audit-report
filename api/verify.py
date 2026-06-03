"""Vercel Python 서버리스 함수: 라벨 이미지 -> Upstage Information Extract 추출 -> KCIA 사전 대조 -> 판정.

- API 키는 환경변수 UPSTAGE_API_KEY 에서만 읽음 (클라이언트 노출 없음).
- 입력: POST JSON { image: <base64>, mime: "image/jpeg" }
- 출력: { ingredients: [...], tokens: [{token,status,std_kr,suggestion,score,candidates}], summary: {...} }
"""
import base64
import json
import os
import re
import unicodedata
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import requests
from rapidfuzz import fuzz, process

IE_URL = "https://api.upstage.ai/v1/information-extraction"
DICT_PATH = Path(__file__).resolve().parent / "ingredients.jsonl"

# ---- 정규화 (verify 파이프라인과 동일 규칙) ----
HYPHENS = "‐‑‒–—―−－ーㅡᅳ"
HYPHEN_RE = re.compile(f"[{HYPHENS}]")
TRAILER_RE = re.compile(r"[\(\[][^)\]]*(?:ppm|%|함량|추출액)[^)\]]*[\)\]]", re.I)
AMOUNT_RE = re.compile(r"\(\s*[\d.,]+\s*(?:%|ppm|mg|㎎|g|ml|㎖|mL|IU|iu|mcg|㎍|µg)[^)]*\)")
WS_RE = re.compile(r"\s+")


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = TRAILER_RE.sub("", s)
    s = HYPHEN_RE.sub("-", s)
    s = s.replace("·", "").replace("ㆍ", "").replace("‧", "")
    s = WS_RE.sub("", s).strip().lower().strip(".,·-")
    return s


def clean_token(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s or "")
    s = AMOUNT_RE.sub("", s)
    s = WS_RE.sub("", s)
    s = re.sub(r"^\[?전성분\]?", "", s)
    s = s.strip("*· .,|")
    if s.count("(") < s.count(")"):
        s = s.rstrip(")")
    return s


class Dictionary:
    def __init__(self, path=DICT_PATH):
        self.records = [json.loads(l) for l in open(path, encoding="utf-8")]
        self.exact, self.alias, self._keys, self._idx = {}, {}, [], []
        for i, r in enumerate(self.records):
            if r["std_kr"]:
                k = normalize(r["std_kr"])
                self.exact.setdefault(k, r)
                self._keys.append(k)
                self._idx.append(i)
            for a in r.get("aliases_kr", []):
                self.alias.setdefault(normalize(a), r)

    def match(self, token, suspect_th=70):
        key = normalize(token)
        res = {"token": token, "status": "UNKNOWN", "std_kr": None,
               "suggestion": None, "score": None, "candidates": []}
        if not key:
            res["status"] = "EMPTY"
            return res
        if key in self.exact:
            res.update(status="PASS", std_kr=self.exact[key]["std_kr"], score=100)
            return res
        if key in self.alias:
            r = self.alias[key]
            res.update(status="ALIAS", std_kr=r["std_kr"], suggestion=r["std_kr"], score=100)
            return res
        hits = process.extract(key, self._keys, scorer=fuzz.ratio, limit=3)
        if hits:
            cands = [(self.records[self._idx[i]]["std_kr"], round(sc, 1)) for _, sc, i in hits]
            res["candidates"] = [{"std_kr": n, "score": sc} for n, sc in cands]
            res["score"] = cands[0][1]
            res.update(status="SUSPECT" if cands[0][1] >= suspect_th else "UNKNOWN",
                       suggestion=cands[0][0])
        return res


DICT = Dictionary()   # 콜드스타트 1회 로드 (warm 재사용)

IE_SCHEMA = {
    "type": "object",
    "properties": {
        "ingredients": {"type": "array", "items": {"type": "string"},
                        "description": ("제품 뒷면 '전성분' 성분들을 배열로. 각 성분이 한 항목. "
                                        "'1,2-헥산다이올'처럼 콤마 포함 성분명은 쪼개지 말 것. "
                                        "인쇄된 원문 그대로(번역/철자교정/표준화 금지). "
                                        "함량 표기(괄호 안 %·ppm·mg 등 수치)는 제외, 이름 속 숫자는 유지.")}},
    "required": ["ingredients"],
}
MAX_IMG_BYTES = 8 * 1024 * 1024   # 8MB 상한


def run_ie(b64, mime, key):
    payload = {"model": "information-extract",
               "messages": [{"role": "user", "content": [
                   {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}]}],
               "response_format": {"type": "json_schema",
                                   "json_schema": {"name": "ingredient_schema", "schema": IE_SCHEMA}}}
    r = requests.post(IE_URL, headers={"Authorization": f"Bearer {key}",
                      "Content-Type": "application/json"}, json=payload, timeout=55)
    r.raise_for_status()
    return json.loads(r.json()["choices"][0]["message"]["content"]).get("ingredients", [])


class handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        self._send(200, {"ok": True, "dict": len(DICT.exact)})

    def do_POST(self):
        try:
            key = os.environ.get("UPSTAGE_API_KEY")
            if not key:
                return self._send(500, {"error": "서버에 UPSTAGE_API_KEY 미설정"})
            length = int(self.headers.get("content-length", 0))
            if length > MAX_IMG_BYTES * 2:
                return self._send(413, {"error": "이미지가 너무 큽니다(8MB 이하)."})
            data = json.loads(self.rfile.read(length) or b"{}")
            b64 = (data.get("image") or "").split(",")[-1]
            if not b64:
                return self._send(400, {"error": "image(base64) 필요"})
            if len(b64) * 3 // 4 > MAX_IMG_BYTES:
                return self._send(413, {"error": "이미지가 너무 큽니다(8MB 이하)."})
            mime = data.get("mime", "image/jpeg")

            ing = run_ie(b64, mime, key)
            tokens = [t for t in (clean_token(x) for x in ing) if t]
            counts = {"PASS": 0, "ALIAS": 0, "SUSPECT": 0, "UNKNOWN": 0, "EMPTY": 0}
            results = []
            for t in tokens:
                m = DICT.match(t)
                counts[m["status"]] = counts.get(m["status"], 0) + 1
                results.append(m)
            self._send(200, {"ingredients": ing, "tokens": results, "summary": counts})
        except requests.HTTPError as e:
            self._send(502, {"error": f"Upstage API 오류: {getattr(e.response,'status_code','')}"})
        except Exception as e:
            self._send(500, {"error": str(e)[:200]})
