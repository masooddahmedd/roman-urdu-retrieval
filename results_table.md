## Baseline (no fix), Recall@5 on test queries

| Retriever | Urdu script | English | Roman Urdu (mean of 3 spellings) | Drop, Urdu to Roman | Variant spread | Queries where spellings disagree |
|---|---|---|---|---|---|---|
| bm25 | 98.5% | 1.0% | 1.0% | 97.5 pts | 0.0 pts | 0.0% |
| openai | 84.0% | 45.5% | 51.5% | 32.5 pts | 1.0 pts | 14.5% |
| bge-m3 | 99.0% | 80.5% | 40.0% | 59.0 pts | 1.5 pts | 9.0% |
| e5 | 98.0% | 16.5% | 68.8% | 29.2 pts | 2.0 pts | 12.5% |

## Fixes

| Retriever | Fix | Roman Recall@5 | Roman MRR@10 | Variant spread | Disagree |
|---|---|---|---|---|---|
| bm25 | none | 1.0% | 0.005 | 0.0 pts | 0.0% |
| bm25 | normalize | 0.7% | 0.004 | 0.5 pts | 0.5% |
| bm25 | transliterate | 96.3% | 0.880 | 0.5 pts | 2.0% |
| openai | none | 51.5% | 0.357 | 1.0 pts | 14.5% |
| openai | normalize | 37.2% | 0.255 | 3.5 pts | 8.5% |
| openai | transliterate | 79.5% | 0.693 | 1.5 pts | 4.5% |
| openai | hybrid | 91.5% | 0.819 | 1.0 pts | 1.5% |
| bge-m3 | none | 40.0% | 0.291 | 1.5 pts | 9.0% |
| bge-m3 | normalize | 35.2% | 0.271 | 2.0 pts | 7.5% |
| bge-m3 | transliterate | 94.7% | 0.872 | 0.5 pts | 0.5% |
| bge-m3 | hybrid | 96.5% | 0.816 | 1.5 pts | 2.5% |
| e5 | none | 68.8% | 0.537 | 2.0 pts | 12.5% |
| e5 | normalize | 63.8% | 0.495 | 2.0 pts | 8.5% |
| e5 | transliterate | 97.2% | 0.883 | 0.5 pts | 0.5% |
| e5 | hybrid | 96.3% | 0.847 | 1.0 pts | 2.0% |
