# 🎧 Article → Audio

Paste a link to any news article or blog post and listen to it instead of reading it.

- **Natural voices.** Uses [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M), one of the best-sounding open text-to-speech models, with 11 US and UK voices.
- **Starts fast.** Audio streams into the player and usually starts within about 2 seconds. Generation then runs ahead of playback.
- **Fully local.** Speech is generated on your machine. The only network request is fetching the article itself. No API keys or accounts.
- **Simple.** One Python file and one HTML page.

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/liorshahverdi/article-audio.git
cd article-audio
uv run uvicorn app:app --port 8765
```

Open http://127.0.0.1:8765. On the first run, the model files (~340MB) download automatically into `models/`.

## Features

- Readable article text extraction via [trafilatura](https://github.com/adbar/trafilatura)
- Voice picker and playback speed control (0.9–2x), remembered between visits
- Download as MP3
- Finished articles are cached in memory, so replaying, seeking, and downloading them is instant

## How it works

1. `GET /api/article?url=…` fetches the page and extracts the title and main text.
2. `GET /api/audio/{id}?voice=…` splits the text into chunks and synthesizes them with Kokoro (via [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx)), encoding to MP3 as it goes. The audio is streamed back as it's generated. The first chunks are deliberately short so playback starts quickly.

On an Apple Silicon Mac (CPU), generation runs at roughly 3–4x real-time.

## Limitations

- Paywalled pages and pages that render their text with JavaScript can't be extracted.
- Some sites block automated requests (Wikipedia currently returns 403).
- While an article is still generating, you can't seek past the point generation has reached. Once it finishes, it's fully seekable.
- English voices only.
- Intended to run locally. There's no authentication, and the server will fetch any URL it's given.

## License

MIT. Kokoro model weights are Apache-2.0.
