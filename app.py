"""Article -> audio. Extracts readable text from a URL and streams local Kokoro TTS as MP3."""

import asyncio
import hashlib
import re
from collections import OrderedDict
from pathlib import Path

import httpx
import lameenc
import numpy as np
import trafilatura
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response, StreamingResponse
from kokoro_onnx import Kokoro

ROOT = Path(__file__).parent
STATIC = ROOT / "static"
DEFAULT_VOICE = "af_heart"

MODELS = ROOT / "models"
MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
MODEL_FILES = ["kokoro-v1.0.onnx", "voices-v1.0.bin"]


def ensure_models() -> None:
    """Download the Kokoro model files on first run (~340MB total)."""
    MODELS.mkdir(exist_ok=True)
    for name in MODEL_FILES:
        dest = MODELS / name
        if dest.exists():
            continue
        print(f"Downloading {name} (first run only)...", flush=True)
        tmp = dest.with_suffix(".part")
        with httpx.stream("GET", MODEL_URL + name, follow_redirects=True, timeout=60) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
        tmp.rename(dest)


ensure_models()
app = FastAPI()
tts = Kokoro(str(MODELS / MODEL_FILES[0]), str(MODELS / MODEL_FILES[1]))
tts.create("Ready.", voice=DEFAULT_VOICE, lang="en-us")  # warm up so the first request is fast

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"
)
http = httpx.AsyncClient(
    headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"},
    follow_redirects=True,
    timeout=15,
)


class LRU(OrderedDict):
    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def put(self, key, value):
        self[key] = value
        self.move_to_end(key)
        while len(self) > self.size:
            self.popitem(last=False)


articles: LRU = LRU(50)  # id -> {title, text}
audio_cache: LRU = LRU(20)  # (id, voice) -> mp3 bytes (only complete renders)


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/article")
async def article(url: str = Query(...)):
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    art_id = hashlib.sha1(url.encode()).hexdigest()[:16]
    if art_id in articles:
        return {"id": art_id, **articles[art_id]}

    try:
        r = await http.get(url)
        if r.status_code == 403:  # some sites (e.g. Wikipedia) reject spoofed browser UAs
            r = await http.get(url, headers={"User-Agent": "article-audio/1.0 (personal reader)"})
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Couldn't fetch page: {e}")

    def parse(html: str):
        opts = dict(include_comments=False, include_tables=False)
        text = trafilatura.extract(html, favor_precision=True, **opts)
        if not text or len(text) < 500:  # unusual layouts: retry being less strict
            text = trafilatura.extract(html, favor_recall=True, **opts) or text
        meta = trafilatura.extract_metadata(html)
        return text, (meta.title if meta and meta.title else None)

    text, title = await asyncio.to_thread(parse, r.text)
    if not text or len(text) < 200:
        raise HTTPException(422, "Couldn't find article text on that page (it may be paywalled or JS-rendered).")

    data = {"title": title or url, "text": text}
    articles.put(art_id, data)
    return {"id": art_id, **data}


@app.get("/api/audio/{art_id}")
async def audio(art_id: str, voice: str = DEFAULT_VOICE, download: bool = False):
    art = articles.get(art_id)
    if not art:
        raise HTTPException(404, "Unknown article; load it again.")
    if voice not in VOICES:
        voice = DEFAULT_VOICE

    key = (art_id, voice)
    filename = "".join(c for c in art["title"] if c.isalnum() or c in " -_")[:80].strip() or "article"
    disp = "attachment" if download else "inline"
    headers = {"Content-Disposition": f'{disp}; filename="{filename}.mp3"'}

    if key in audio_cache:  # full file -> seekable
        return Response(audio_cache[key], media_type="audio/mpeg", headers=headers)

    body = art["text"]
    if not body[:200].lower().startswith(art["title"].lower()[:40]):  # avoid reading the title twice
        body = f"{art['title']}.\n\n{body}"
    pieces = split_text(body)
    lang = "en-gb" if voice.startswith("b") else "en-us"

    async def gen():
        # A producer synthesizes chunks back-to-back (~4x realtime) into a queue
        # while we stream them out, so it stays ahead of playback. Cancelled if
        # the client disconnects (e.g. switches voice).
        q: asyncio.Queue = asyncio.Queue()
        enc = lameenc.Encoder()
        enc.set_bit_rate(64)
        enc.set_in_sample_rate(24000)
        enc.set_channels(1)
        enc.set_quality(2)

        def render(text: str) -> bytes:
            samples, _ = tts.create(text, voice=voice, lang=lang)
            pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes()
            return bytes(enc.encode(pcm))

        async def produce():
            for p in pieces:
                await q.put(await asyncio.to_thread(render, p))
            await q.put(bytes(enc.flush()))
            await q.put(None)

        producer = asyncio.create_task(produce())
        out = []
        try:
            while (data := await q.get()) is not None:
                out.append(data)
                yield data
            audio_cache.put(key, b"".join(out))
        finally:
            producer.cancel()

    return StreamingResponse(gen(), media_type="audio/mpeg", headers=headers)


def split_text(text: str, size: int = 400) -> list[str]:
    """Sentence-sized chunks. The opening is split at clause boundaries too, so
    the first audio is ready in about a second."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    chunks: list[str] = []
    # Fast start: the first few chunks are short (clauses, or ~12-word runs)
    # so the first audio is quick; by then the producer is ahead of playback.
    while sentences and len(chunks) < 4:
        for clause in re.split(r"(?<=[,;:])\s+", sentences.pop(0)):
            words = clause.split()
            chunks += [" ".join(words[i : i + 12]) for i in range(0, len(words), 12)]
    cur = ""
    for s in sentences:
        if cur and len(cur) + len(s) > size:
            chunks.append(cur)
            cur = ""
        cur = f"{cur} {s}" if cur else s
    if cur:
        chunks.append(cur)
    return chunks


VOICE_NAMES = {
    "af_heart": "Heart (US, female)",
    "af_bella": "Bella (US, female)",
    "af_nicole": "Nicole (US, female, soft)",
    "af_sarah": "Sarah (US, female)",
    "am_michael": "Michael (US, male)",
    "am_fenrir": "Fenrir (US, male)",
    "am_puck": "Puck (US, male)",
    "bf_emma": "Emma (UK, female)",
    "bf_isabella": "Isabella (UK, female)",
    "bm_george": "George (UK, male)",
    "bm_fable": "Fable (UK, male)",
}
VOICES = {v: VOICE_NAMES[v] for v in VOICE_NAMES if v in set(tts.get_voices())}


@app.get("/api/voices")
async def voices():
    return [{"id": k, "label": v} for k, v in VOICES.items()]
