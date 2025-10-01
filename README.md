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

Dia1.6b adapted to Portuguese. The model lost non-verbal signals and has minor flaws due to the mixed database between English and Brazilian Portuguese. However, it is a great starting point if you want to train your own fork.

---

## 📜 License

Apache 2.0 — matches the original project by Nari Labs.
