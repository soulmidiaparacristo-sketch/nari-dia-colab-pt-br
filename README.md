# 🎤 Nari Dia TTS — Colab Edition (Float32 & T4-Compatible)

Uma versão exclusiva para Colab do modelo Dia da Nari Labs, projetada especificamente para execução perfeita em GPUs T4 dentro do Google Colab. Esta bifurcação elimina problemas de compatibilidade com AMP e Gradio v5 e inclui um notebook pré-configurado para geração de voz com um clique.

---

## 🚀 Quickstart (Run on Google Colab)

Clique abaixo para iniciar o notebook mais recente:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/soulmidiaparacristo-sketch/nari-dia-colab-pt-br/blob/main/Nari_Dia_Colab.ipynb)

---

## ✅ Por que este fork?

Esta versão foi criada para:

- Execute de forma confiável no **Google Colab** usando **GPUs T4**
- Evite erros de dtype (`bfloat16` vs `float32`) desabilitando o AMP
- Garanta que a **IU do Gradio funcione com a versão 5+** (sem argumentos obsoletos)
- Forneça um notebook `.ipynb` limpo e reproduzível, projetado exclusivamente para o Google Colab

---

## 📄 O que está incluído

| Arquivo | Finalidade |
|------|---------|
| `app.py` | Limpo e modificado para Colab |
| `Nari_Dia_Colab.ipynb` | Caderno pré-configurado para inferência |
| `README.md` | Este arquivo |

---

Dia1.6b adaptado para o português. O modelo perdeu sinais não verbais e apresenta pequenas falhas devido ao banco de dados misto entre inglês e português brasileiro. No entanto, é um ótimo ponto de partida se você quiser treinar seu próprio fork e testa em colab, alem de usar o modelo Dia pronto com vozes naturais com pequenos ajuste de escrita romanizados.
agradecimentos: https://github.com/arcaneum/nari-dia-colab & https://huggingface.co/Alissonerdx/Dia1.6-pt_BR-v1


---

## 📜 License

Apache 2.0 — matches the original project by Nari Labs.
