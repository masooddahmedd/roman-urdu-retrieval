# roman-urdu-retrieval

A Roman Urdu retrieval benchmark: how much does it hurt semantic search and RAG when someone types an Urdu question in Latin letters (and spells it however they like) instead of Urdu script. We measure BM25, bge-m3, multilingual-e5-large-instruct and OpenAI text-embedding-3-large on 200 test questions, then test three fixes. The short version: the drop is large, spelling variation is a smaller problem than script, and transliterating the query to Urdu script before searching fixes most of it.

![Roman Urdu Recall@5 before and after each fix](fixes_chart.png)

## Results

200 test questions, each asked in Urdu script, in English and in three Roman Urdu spellings. The pool is 3,674 Wikipedia passages (1,674 Urdu, 2,000 English). The right answer is always one Urdu passage. Recall@5 is the share of questions where that passage is in the top 5. Everything comes from `run_benchmark.py` (see `results_table.md` and `results.json`).

| Retriever | Urdu script | English | Roman Urdu (mean of 3 spellings) | Drop, Urdu to Roman | Questions where the 3 spellings disagree |
| --- | --- | --- | --- | --- | --- |
| BM25 | 98.0% | 1.0% | 1.7% | 96.3 pts | 0.5% |
| OpenAI text-embedding-3-large | 84.0% | 46.0% | 50.8% | 33.2 pts | 13.5% |
| bge-m3 | 99.0% | 81.5% | 44.5% | 54.5 pts | 18.0% |
| multilingual-e5-large-instruct | 98.0% | 16.5% | 70.3% | 27.7 pts | 14.0% |

Roman Urdu Recall@5 after each fix (same 200 questions, mean of the 3 spellings, 95% bootstrap interval over questions in brackets):

| Retriever | No fix | Spelling rules | LLM transliteration | Hybrid |
| --- | --- | --- | --- | --- |
| BM25 | 1.7% [0.2, 3.5] | 1.0% [0.0, 2.5] | 94.7% [91.5, 97.5] | n/a |
| OpenAI | 50.8% [44.5, 57.3] | 40.2% [34.0, 46.3] | 79.8% [74.3, 84.8] | 92.7% [89.0, 96.0] |
| bge-m3 | 44.5% [38.3, 50.7] | 39.5% [33.3, 45.8] | 96.3% [93.8, 98.5] | 98.0% [95.8, 99.5] |
| e5-instruct | 70.3% [64.7, 76.3] | 66.2% [60.2, 72.5] | 97.5% [95.2, 99.3] | 96.5% [94.0, 98.7] |

What the numbers say:

- **Script is the big problem.** Every dense model loses between 28 and 55 points of Recall@5 when the same question moves from Urdu script to Roman Urdu. BM25 loses almost everything, which is expected: it matches letters, and a Latin-script query shares none with an Urdu-script passage.
- **Transliteration fixes most of it.** Converting the Roman Urdu query to Urdu script with gpt-4o-mini before searching brings bge-m3 to 96.3% (Urdu script is 99.0%), e5 to 97.5% (98.0%) and BM25 to 94.7% (98.0%). OpenAI embeddings get to 79.8%, which is 95% of their own Urdu-script score of 84.0%.
- **The hybrid** (rank fusion of the original query, the transliterated query and BM25 on the transliterated query) is best for OpenAI (92.7%, above its Urdu-script baseline) and bge-m3 (98.0%). For e5 it is slightly below transliteration alone (96.5% vs 97.5%, inside the noise).
- **Rule-based spelling normalization made things worse.** Four rules survived tuning on the dev questions (fold repeated letters, `kia`/`kiya` to `kya`, `ai`/`ay`/`ae` to `e`, `q` to `k`). Applied to the queries they cost 4 to 11 points of Recall@5 and raised the spread between spellings for the dense models. The normalized text is not how anyone writes, and the embedding models have seen natural spellings.
- **Spelling variation is smaller than script, but real.** The gap in Recall@5 between the best and worst of the 3 spellings is only 0.5 to 1.5 points. That hides the fact that for 13.5% to 18% of questions the same question is found with one spelling and missed with another. Transliteration cuts that to between 1.0% and 6.0% for the dense models (1.0% for e5, 2.5% for bge-m3, 6.0% for OpenAI).
- **Why English queries look so different across models.** The pool has 2,000 English passages on similar topics, so an English or Roman Urdu query can land on English passages instead of the Urdu one (`language_bias.json`). In the top 5 for Roman Urdu queries, English passages make up 73% for OpenAI, 73 to 75% for bge-m3 and 33 to 35% for e5. For English queries e5 returns 95% English passages, which is why its English Recall@5 is only 16.5%.

### How much of the drop is the English distractors?

Part of it. Removing the English passages from the candidate pool (`language_bias.py`) gives Roman Urdu Recall@5 of 75.8% for OpenAI (Urdu script 86.5%), 62.3% for bge-m3 (99.0%) and 77.2% for e5 (98.0%). So the gap shrinks to between 11 and 37 points, but it does not go away. Roman Urdu is hard for these models even without a bilingual pool.

## How it works

1. `fetch_corpus.py` pulls Urdu and English Wikipedia articles on Pakistani topics and cuts them into passages of 60 to 170 words (up to 4 per article).
2. `make_queries.py` has gpt-4o write one specific question per sampled Urdu passage, plus the English version and three Roman Urdu spellings. 300 questions, split 100 dev and 200 test.
3. `retrievers.py` runs BM25, bge-m3 and e5 (official ONNX exports, CPU) and OpenAI embeddings (1,024 dimensions). Embeddings are cached by text.
4. `fixes.py` holds the three fixes. The spelling rules were chosen on the dev questions only, and every number above is on the test questions.
5. `run_benchmark.py` computes Recall@5, MRR@10, the spread across spellings and the share of questions where the spellings disagree.

## The query set

`queries.jsonl` and `queries.csv` (CC BY 4.0, see `LICENSE-DATA`): 300 questions, each with the Urdu question, the English question, three Roman Urdu spellings and the id of the Urdu passage that answers it. The Wikipedia passage text is not included (it is CC BY-SA). Run `fetch_corpus.py` to rebuild it; the ids are stable as long as the article revisions are.

The questions are LLM-drafted. A native Urdu speaker is spot-checking a 50-question sample (`review_sample.csv`) for natural phrasing and spelling. Status of that check: **pending**. Treat the Roman Urdu spellings as plausible, not as verified typing habits, until that is filled in.

## Run it

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # .venv/bin/python on Linux/macOS
python fetch_corpus.py        # rebuilds data/passages.jsonl
python run_benchmark.py       # downloads the ONNX models (about 4.5 GB) on first run
```

The transliterations (`cache/translit.json`) and the query drafts are cached in the repo, so those steps need no API key. Passage embeddings are not committed (they are derived from CC BY-SA text), so the first run re-embeds the corpus: about an hour on a laptop CPU for the two ONNX models, plus an OpenAI key for the OpenAI retriever (copy `.env.example` to `.env`). Later runs reuse the local cache. Tests: `python -m pytest -q`.

## Limitations

- **LLM-drafted questions.** Some Roman Urdu spellings differ by more than spelling (a different word choice, for example). Some questions say "this passage" or "here", which makes them easier than a real search query. The native-speaker check is pending.
- **One gold passage per question.** Other passages may also answer it, so Recall@5 is a lower bound.
- **Mixed-language pool.** English distractors inflate the Roman Urdu drop (see above). A purely Urdu pool is the other half of the picture.
- **200 test questions.** The intervals above are wide enough that differences of a few points between models should not be read as a ranking.
- **Urdu corpus is 1,674 passages, not 2,000.** Urdu Wikipedia ran out of long enough articles on these topics.
- **CPU models truncate at 400 tokens** and OpenAI embeddings use 1,024 dimensions. A different setup could move numbers by a point or two.
- **Transliteration uses an LLM at query time.** That is one extra model call per query (600 short gpt-4o-mini calls for this benchmark), which is latency and cost a plain embedding model does not have. I did not measure either.
- **Wikipedia only.** Chat messages and social media text, where Roman Urdu is most common, would likely be harder.

## Credits

- Starting point for the question: [roman-urdu-rag-bias-audit](https://github.com/malahilghauri-design/roman-urdu-rag-bias-audit) by malahilghauri-design, which measured how little the result lists overlap across spellings with one multilingual model. This repo adds gold-passage recall, the cross-script case, other retrievers and fixes.
- Models: [bge-m3](https://huggingface.co/BAAI/bge-m3) (BAAI), [multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct) (Microsoft/intfloat), OpenAI text-embedding-3-large and gpt-4o / gpt-4o-mini.
- Text: Urdu and English Wikipedia, CC BY-SA 4.0.
- Related work on Roman Urdu normalization and transliteration, for example [A Clustering Framework for Lexical Normalization of Roman Urdu](https://arxiv.org/abs/2004.00088) and [Low-Resource Transliteration for Roman-Urdu and Urdu](https://arxiv.org/abs/2503.21530). I did not find an existing benchmark that measures gold-passage recall across embedding models for Roman Urdu, but my search was quick, so I do not claim this is the first.

Code is MIT (`LICENSE`). The query set is CC BY 4.0 (`LICENSE-DATA`).
