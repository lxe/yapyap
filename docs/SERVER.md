# Server

`yapyap-server` runs Whisper and serves the OpenAI transcription API on port 8122.

| Hardware | Engine | Default model |
|---|---|---|
| Intel GPU (Arc, Iris Xe, Core Ultra) | [OpenVINO](https://github.com/openvinotoolkit/openvino.genai) | `OpenVINO/whisper-large-v3-turbo-fp16-ov` |
| NVIDIA GPU | [whisper.cpp](https://github.com/ggml-org/whisper.cpp) with CUDA | `large-v3-turbo-q8_0` |
| CPU | whisper.cpp | `large-v3-turbo-q8_0` |

## Docker

The [README](../README.md#install) has the commands. The first start downloads the model, about 1.6 GB, into the `models` volume. `curl localhost:8122/health` returns `{"status": "ok"}` once the server is ready.

To use a different model, set `YAPYAP_MODEL` in `.env`. Intel takes a Hugging Face repo with an OpenVINO export, such as `OpenVINO/whisper-small-fp16-ov`. NVIDIA and CPU take a [whisper.cpp model name](https://huggingface.co/ggerganov/whisper.cpp), such as `small-q8_0`. Smaller models are faster, especially on CPU.

## Without Docker

```bash
# Intel GPU (needs Intel's GPU compute runtime)
uv tool install "yapyap[openvino] @ git+https://github.com/lxe/yapyap"
yapyap-server --engine openvino

# NVIDIA GPU (builds whisper.cpp against your CUDA toolkit)
GGML_CUDA=1 uv tool install --no-binary-package pywhispercpp "yapyap[whispercpp] @ git+https://github.com/lxe/yapyap"
yapyap-server --engine whispercpp
```

The server listens on 127.0.0.1 unless you pass `--host 0.0.0.0`. `yapyap-server --help` lists the options. Each option can also be set with an environment variable: `YAPYAP_ENGINE`, `YAPYAP_MODEL`, `YAPYAP_DEVICE`, `YAPYAP_HOST`, `YAPYAP_PORT`, `YAPYAP_TOKEN`.

## Benchmark

`yapyap bench` sends 16 kHz WAV files to the server a few times each, with streaming on, and prints the median time to the first words and to the full transcript:

```bash
yapyap bench --server http://localhost:8122 --token "$YAPYAP_TOKEN" speech-05s.wav speech-10s.wav speech-20s.wav
```

## API

```bash
curl localhost:8122/v1/audio/transcriptions \
  -H "Authorization: Bearer $YAPYAP_TOKEN" \
  -F file=@speech.wav -F stream=true
```

This is OpenAI's [transcription endpoint](https://platform.openai.com/docs/api-reference/audio/createTranscription). It takes these form fields:

| Field | Meaning |
|---|---|
| `file` | Audio. 16 kHz mono WAV is fastest; the Docker images accept any format through ffmpeg. |
| `language` | Language code such as `en`. Without it, the server detects the language. |
| `prompt` | Words or a sample sentence that guide spelling and mixed languages. |
| `stream` | `true` streams the text as it's decoded, as `transcript.text.delta` events, then a final `transcript.text.done`. |
| `response_format` | `json` (the default) or `text`. |

`GET /health` doesn't need the token. `GET /v1/models` returns the name of the loaded model.
