---
publish_at: 2026-10-06 16:00 +05:00
status: draft
media: C:\Users\LENOVO\Documents\work\projects\selfProjects\02-roman-urdu-retrieval\spelling_example.png
---
1/ Same question, three Roman Urdu spellings: bge-m3 ranked the right passage 52nd, 4th and 34th. I measured how much Roman Urdu hurts embedding search and RAG on 200 questions and 4 retrievers. Typing in Latin letters cost bge-m3 54 points of Recall@5.

2/ Recall@5, Urdu script to Roman Urdu: bge-m3 99% to 44.5%, OpenAI embeddings 84% to 50.8%, multilingual-e5 98% to 70.3%. BM25 fell to about 0, because the query shares no letters with an Urdu-script passage.

3/ Spelling normalization rules made it worse, by 4 to 11 points. What worked: have gpt-4o-mini turn the query into Urdu script before searching. bge-m3 went back to 96.3%. A hybrid with BM25 took OpenAI to 92.7%.

4/ Caveat: part of the drop is my pool also holding English passages (Roman Urdu looks like English to these models). Without them the gap shrinks to 11 to 37 points but stays. Questions are LLM-drafted, [CHECK: native-speaker review of a 50-question sample done?].

5/ Repo with code and results: https://github.com/masooddahmedd/roman-urdu-retrieval
Query set (CC BY): [HF dataset URL]
Demo, type your own spelling: [HF Space URL]
