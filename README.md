# 🎤 Nari Dia TTS — Colab Edition (Float32 & T4-Compatible)

A Colab-only version of the Nari Labs Dia model, specifically designed for seamless execution on T4 GPUs within Google Colab. This fork removes compatibility issues with AMP and Gradio v5, and includes a pre-configured notebook for one-click voice generation.

---

## 🚀 Quickstart (Run on Google Colab)

Click below to launch the latest notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/soulmidiaparacristo-sketch/nari-dia-colab-pt-br/blob/main/Nari_Dia_Colab.ipynb)

---

## ✅ Why This Fork?

This version was created to:

- Run reliably in **Google Colab** using **T4 GPUs**
- Avoid dtype errors (`bfloat16` vs `float32`) by disabling AMP
- Ensure **Gradio UI works with v5+** (no deprecated args)
- Provide a clean, reproducible `.ipynb` notebook designed exclusively for Google Colab

---

## 📄 What's Included

| File | Purpose |
|------|---------|
| `app.py` | Cleaned up and modified for Colab |
| `Nari_Dia_Colab.ipynb` | Pre-configured notebook for inference |
| `README.md` | This file |

---

## 🧠 Model Background

Dia is a 1.6B parameter text-to-speech model by [Nari Labs](https://github.com/nari-labs/dia). It supports:
- Text-to-dialogue generation with `[S1]`, `[S2]` speaker tags
- Non-verbal cues like `(laughs)`, `(coughs)`
- Voice conditioning using uploaded audio

Model weights are loaded via:
```python
Dia.from_pretrained("nari-labs/Dia-1.6B", device="cuda")
```

---

## 📁 Local Use Not Supported

This repository is intentionally designed **only for use within Google Colab** to ensure a reliable and reproducible environment.  

If you would like to use Dia locally or explore ongoing development, please refer to the [original Nari Labs repository](https://github.com/nari-labs/dia).

---

## 🙋 Maintainer

Created by **Anil Clifford**  
→ Experimental projects via [@arcaneum](https://github.com/arcaneum)  
→ Public releases via [EdenDigitalUK](https://github.com/EdenDigitalUK)  
→ Professional work at [Eden Digital](https://www.edendigital.io)  
→ [LinkedIn](https://www.linkedin.com/in/anilcliff/) · [Twitter/X](https://x.com/anil_clifford)

---

## 📜 License

Apache 2.0 — matches the original project by Nari Labs.
