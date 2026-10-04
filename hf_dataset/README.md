---
license: cc-by-4.0
language:
- ur
- en
pretty_name: Roman Urdu Retrieval Queries
task_categories:
- text-retrieval
tags:
- roman-urdu
- urdu
- retrieval
- rag
- benchmark
- multilingual
size_categories:
- n<1K
configs:
- config_name: queries
  data_files: queries.jsonl
- config_name: corpus
  data_files: corpus.jsonl
---

# Roman Urdu Retrieval Queries

300 questions with a known answer passage, each written four ways: Urdu script, English and three
Roman Urdu spellings. It is the query set for the
[roman-urdu-retrieval benchmark](https://github.com/masooddahmedd/roman-urdu-retrieval), which
measures how much Roman Urdu spelling and script hurt embedding search and RAG.

## Files

| File | What it is | License |
| --- | --- | --- |
| `queries.jsonl`, `queries.csv` | 300 questions (100 dev, 200 test) | CC BY 4.0 |
| `review_sample.csv` | The 50 questions sampled for native-speaker review | CC BY 4.0 |
| `corpus.jsonl` | 3,674 Wikipedia passages (1,674 Urdu, 2,000 English) the answers come from | CC BY-SA 4.0 (Wikipedia contributors) |

The CC BY license covers the queries. `corpus.jsonl` is Wikipedia text and stays CC BY-SA 4.0;
each row has the article title and page id for attribution.

## Fields

`queries`: `id`, `split` (dev or test), `gold_id` (id of the Urdu passage that answers it),
`urdu`, `english`, `roman_1`, `roman_2`, `roman_3`.
`corpus`: `id`, `lang`, `title`, `pageid`, `text`.

## How it was built

- Passages: Urdu and English Wikipedia articles on Pakistani topics, cut into passages of 60 to 170
  words.
- Questions: for each of 300 sampled Urdu passages, gpt-4o wrote one specific question in Urdu
  script, the same question in English, and three Roman Urdu spellings as different people might
  type it. They are LLM-drafted.
- Human review: a native Urdu speaker is checking the 50-question sample in `review_sample.csv`
  for natural phrasing and spelling. Status: **[pending: update when done]**.

## Known limitations

- The Roman Urdu spellings are LLM-written. Some differ by more than spelling (a different word
  choice), and they are not samples of real typing.
- Some questions say "this passage" or "here", which makes them easier than a real search query.
- One gold passage per question. Other passages may also answer it.
- Wikipedia text only. Chat and social media Roman Urdu is likely harder.

## Results this was used for

Recall@5 on the 200 test questions (full tables in the repo): moving from Urdu script to Roman Urdu
cost bge-m3 54.5 points (99.0% to 44.5%), OpenAI text-embedding-3-large 33.2 points and
multilingual-e5-large-instruct 27.7 points. Transliterating the query to Urdu script before
searching recovered most of it (bge-m3 96.3%).

## Citation

```
Masood Ahmed. Roman Urdu Retrieval Queries. 2026.
https://github.com/masooddahmedd/roman-urdu-retrieval
```
