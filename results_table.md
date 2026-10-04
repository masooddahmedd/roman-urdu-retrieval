## Baseline (no fix), Recall@5 on test queries

| Retriever | Urdu script | English | Roman Urdu (mean of 3 spellings) | Drop, Urdu to Roman | Variant spread | Queries where spellings disagree |
|---|---|---|---|---|---|---|
| bm25 | 98.0% | 1.0% | 1.7% | 96.3 pts | 0.5 pts | 0.5% |
| openai | 84.0% | 46.0% | 50.8% | 33.2 pts | 1.5 pts | 13.5% |
| bge-m3 | 99.0% | 81.5% | 44.5% | 54.5 pts | 1.5 pts | 18.0% |
| e5 | 98.0% | 16.5% | 70.3% | 27.7 pts | 1.0 pts | 14.0% |

## Fixes

| Retriever | Fix | Roman Recall@5 | Roman MRR@10 | Variant spread | Disagree |
|---|---|---|---|---|---|
| bm25 | none | 1.7% | 0.007 | 0.5 pts | 0.5% |
| bm25 | normalize | 1.0% | 0.005 | 0.0 pts | 0.0% |
| bm25 | transliterate | 94.7% | 0.855 | 2.0 pts | 2.5% |
| openai | none | 50.8% | 0.347 | 1.5 pts | 13.5% |
| openai | normalize | 40.2% | 0.280 | 2.5 pts | 16.0% |
| openai | transliterate | 79.8% | 0.676 | 0.5 pts | 6.0% |
| openai | hybrid | 92.7% | 0.800 | 2.0 pts | 3.0% |
| bge-m3 | none | 44.5% | 0.333 | 1.5 pts | 18.0% |
| bge-m3 | normalize | 39.5% | 0.299 | 4.5 pts | 14.5% |
| bge-m3 | transliterate | 96.3% | 0.887 | 0.5 pts | 2.5% |
| bge-m3 | hybrid | 98.0% | 0.828 | 1.0 pts | 1.5% |
| e5 | none | 70.3% | 0.550 | 1.0 pts | 14.0% |
| e5 | normalize | 66.2% | 0.524 | 3.5 pts | 12.0% |
| e5 | transliterate | 97.5% | 0.888 | 1.0 pts | 1.0% |
| e5 | hybrid | 96.5% | 0.839 | 1.0 pts | 1.0% |
