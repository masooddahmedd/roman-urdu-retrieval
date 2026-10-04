# Renders one real test question in its three Roman Urdu spellings, with the rank each retriever
# gave the right passage. This is the image for the X thread: same question, different results.
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from retrievers import build_retrievers
from run_benchmark import ROOT, gold_ranks, load_data

QUESTION_ID = sys.argv[1] if len(sys.argv) > 1 else "q236"
passages, queries = load_data()
q = next(x for x in queries if x["id"] == QUESTION_ID)
gold = [{p["id"]: i for i, p in enumerate(passages)}[q["gold_id"]]]
names = {"bge-m3": "bge-m3", "openai": "OpenAI embeddings", "e5": "multilingual-e5"}
retrievers = build_retrievers(passages, list(names))

rows = []
for form in ("roman_1", "roman_2", "roman_3"):
    ranks = [int(gold_ranks(retrievers[n].scores([q[form]]), gold)[0]) for n in names]
    rows.append([q[form]] + [str(r) for r in ranks])

fig, ax = plt.subplots(figsize=(11, 3.2))
ax.axis("off")
table = ax.table(
    cellText=rows,
    colLabels=["Same question, three spellings"]
    + [f"{n}\nrank of right answer" for n in names.values()],
    loc="center",
    cellLoc="center",
    colWidths=[0.46, 0.18, 0.18, 0.18],
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.4)
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if r == 0:
        cell.set_facecolor("#f0f0f0")
        cell.set_text_props(weight="bold")
    elif c > 0:
        rank = int(rows[r - 1][c])
        cell.set_facecolor("#d9f2d9" if rank <= 5 else "#f8d7d7")
    if c == 0 and r > 0:
        cell.set_text_props(ha="left")
        cell._loc = "left"
fig.tight_layout()
fig.savefig(ROOT / "spelling_example.png", dpi=170)
print(QUESTION_ID, rows)
