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
| `review_sample.csv` | 50 sampled questions with an AI review (flags, notes, corrected spellings) | CC BY 4.0 |
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
- Questions: for each of 300 sampled Urdu passages, gpt-4o wrote one specific question in Urdu script and in English. A second gpt-4o step then wrote three Roman Urdu spellings of that same Urdu sentence under a spelling-only rule (same words, same order); variant sets that differ in word count or share too few words after normalization were rejected and retried. Questions that refer to "this passage" were dropped. All of it is LLM-written.
- Review: **no native-speaker review has been done.** An AI review (Claude, not a native speaker) of the 50-question sample in `review_sample.csv` passed 42 and flagged 8 (spelling slips such as a mangled place name, and one ungrammatical source question). Corrected spellings are in the `fixed_roman_*` columns. An earlier version of the set mixed spelling changes with word changes, and was regenerated; the archived first version is in the GitHub repo under `v1/`.

## Known limitations

- The Roman Urdu spellings are LLM-written and not native-speaker reviewed. They are GPT-4o's guess at how people type, not samples of real typing.
- A few questions are time-sensitive or oddly worded (one source question has a grammar error).
- One gold passage per question. Other passages may also answer it.
- Wikipedia text only. Chat and social media Roman Urdu is likely harder.

## Results this was used for

Recall@5 on the 200 test questions (full tables in the repo): moving from Urdu script to Roman Urdu
cost bge-m3 59.0 points (99.0% to 40.0%), OpenAI text-embedding-3-large 32.5 points and
multilingual-e5-large-instruct 29.2 points. Transliterating the query to Urdu script before
searching recovered most of it (bge-m3 94.7%).

## Citation

```
Masood Ahmed. Roman Urdu Retrieval Queries. 2026.
https://github.com/masooddahmedd/roman-urdu-retrieval
```
