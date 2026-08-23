# Versilume

> **Cross-Lingual Poem-to-Visual Synthesis Platform**  
> Developed by **Ishan Chowdhury**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Versilume is an AI-powered system that transforms poems into visually coherent digital artwork. It combines deep multi-agent literary extraction, poetic-meaning-preserving neural translation across 127 languages, iterative prompt refinement, and state-of-the-art diffusion image generation.

---

## Architecture Overview

```
                        ┌──────────────────────────────┐
                        │         Input Poem           │
                        │    (127 Source Languages)    │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │   Google Gemini Translation  │
                        │ (Preserves Poetic Resonance) │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │       EPE Segmentation       │
                        │   (Entity & Emotion Shift)   │
                        └──────────────┬───────────────┘
                                       │
                 ┌─────────────────────┼─────────────────────┐
                 │ (Concurrent Execution per Segment)        │
                 ▼                     ▼                     ▼
      ┌────────────────────┐ ┌────────────────────┐ ┌────────────────────┐
      │   Emotion Agent    │ │   Semantic Agent   │ │   Metaphor Agent   │
      │  (Llama-3.2 via HF)│ │ (Llama-3.2 via HF) │ │ (Llama-3.2 via HF) │
      │ Nuance & Intensity │ │ Theme & Symbology  │ │ Concrete Visuals   │
      └──────────┬─────────┘ └─────────┬──────────┘ └─────────┬──────────┘
                 │                     │                      │
                 └─────────────────────┼──────────────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │    Prompt Construction       │
                        │  (Style, Palette & Lighting) │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │    Gemini MSPR Refinement    │
                        │ (Multi-Stage Refinement Loop)│
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │    Diffusion Image Gen       │
                        │ FLUX.1-schnell / SDXL /      │
                        │     Pollinations Fallback    │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │   Visual Art Presentation    │
                        └──────────────────────────────┘
```

---

## Core Features

- **127-Language Translation**: Nuanced poetic translation powered by Google Gemini, preserving metaphor and cadence rather than literal word-for-word translation.
- **Tri-Agent NLP Analysis**: Three specialized Llama-3.2 LLM agents running concurrently:
  - **Emotion Agent**: Identifies refined emotion, subtle tonal nuance, and emotional intensity.
  - **Semantic Agent**: Extracts literary themes, symbolic imagery, and cultural context.
  - **Metaphor Agent**: Resolves figurative poetry and metaphors into paintable visual elements, lighting cues, and curated color palettes.
- **EPE Scene Segmentation**: Breaks long multi-scene poems into visual segments based on entity and emotion shifts.
- **MSPR Prompt Refinement**: Multi-Stage Prompt Refinement loop using Gemini with convergence detection.
- **Resilient Image Generation**: High-fidelity rendering via FLUX.1 / SDXL with an automatic keyless fallback chain to ensure 100% uptime.

---

## Tech Stack

- **Backend**: FastAPI, Uvicorn, Pydantic v2
- **NLP & Embeddings**: spaCy, Transformers (BART, DistilRoBERTa), Sentence-Transformers
- **LLM Orchestration**: Google Gemini 2.0 Flash, Llama-3.2 (via Hugging Face Inference API)
- **Image Generation**: FLUX.1-schnell, Stable Diffusion XL, Pollinations.ai
- **Frontend**: Vanilla HTML5, CSS3, JavaScript (Glassmorphism & dynamic exhibition gallery)

---

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/VijayLabKit/Versilume.git
cd Versilume/poem2image
```

### 2. Configure Environment
```bash
cp .env.example .env
```
Add your API credentials to `.env`:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
HF_API_TOKEN=your_huggingface_token_here
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Launch the Server
```bash
uvicorn app.main:app --reload --port 8000
```
- Interactive API Documentation: `http://localhost:8000/docs`
- Web Interface: Open `versilume.html` in your browser.

---

## API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Service health status and active provider chain |
| `GET` | `/api/v1/languages` | List 127 supported source languages |
| `POST` | `/api/v1/analyze` | Perform translation, multi-agent extraction, and prompt synthesis |
| `POST` | `/api/v1/generate` | Full end-to-end pipeline generating final visual artwork |

---

## Author & Developer

**Ishan Chowdhury**  
GitHub: [@VijayLabKit](https://github.com/VijayLabKit)

---

## License

This project is licensed under the MIT License.
