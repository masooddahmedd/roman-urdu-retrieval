---
title: Roman Urdu retrieval
emoji: 🔎
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# Roman Urdu retrieval demo

Type a question in Roman Urdu (any spelling) and compare embedding search results as typed with the
results after the query is transliterated to Urdu script. Companion to the
[roman-urdu-retrieval benchmark](https://github.com/masooddahmedd/roman-urdu-retrieval).

- Model: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (ONNX, runs on free CPU hardware).
- Passages: 3,674 Wikipedia passages on Pakistani topics (CC BY-SA 4.0), vectors precomputed.
- Transliteration: cached for the benchmark examples. For your own queries you can paste an OpenAI
  key; it is used for that one call and is never stored or logged.
