# AI review of the 50-question sample (second version of the question set; the first version's
# review is in v1/review_labels.py) in review_sample.csv. Written by Claude, who is NOT a native
# Urdu speaker, so this is a second-pass sanity check and not the native-speaker review the
# benchmark ultimately needs. Codes: DRIFT = a Roman Urdu variant changes words, not just spelling;
# LEAK = the question points at "this passage" or "you" and cannot stand alone; AMBIG = ambiguous
# entity, merged question or doubtful content; SPELL = a spelling or grammar slip nobody would type.
import csv
from pathlib import Path

ROOT = Path(__file__).parent

NOTES = {
    "q137": ("no", "DRIFT/SPELL: variant 2 uses Italian for Ataliyi, variant 3 Ataliy Ko Pemaon"),
    "q270": ("no", "SPELL: khetay / ihtay for khitte (fixed)"),
    "q005": ("no", "SPELL: variant 2 Brad Peak changes the name, sr for sar"),
    "q116": ("no", "SPELL: Kuwaita / Kuweta for Quetta in variants 2 and 3 (fixed)"),
    "q225": ("no", "SPELL: tehil / tehail / tehill for tahleel, karsakte joined (fixed)"),
    "q102": ("no", "SPELL: kub for kab, sathe for satah (fixed)"),
    "q042": ("no", "SPELL: variant 3 rakhhi gayee (fixed)"),
    "q227": (
        "no",
        "GRAMMAR: the Urdu source question is ungrammatical (kaisa behtari). Suggest drop",
    ),
}
FIXES = {
    "q137": [
        "",
        "Kis Ataliyi Koh Pemaon nay 31 July 1954 ko K2 ko pehli bar sar kiya?",
        "Kis Ataliyi Koh Pemaon ne 31 July 1954 ko K2 ko pehli baar sar kiya?",
    ],
    "q270": [
        "Tandoori khanon ki khasiyat kis khitte ki rawayat ka tasalsul hai?",
        "",
        "Tandoori khanon ki khasiyat kis khittay ki riwayat ka tasalsul hai?",
    ],
    "q005": [
        "",
        "Broad Peak ko sardiyon mein sar karne waali pehli Polish team mein kon mashhoor koh payma shamil thay?",
        "Broad Peak ko sardiyon mein sar karne wali pehli Polish team mein kon mashhoor koh pehma shamil thay?",
    ],
    "q116": [
        "",
        "Quetta Railway Station Pakistan kay kitnay oonche Railway Stationon mein se aik hai?",
        "Quetta Railway Station Pakistan ke kitne unche Railway Stationon mein se aik hai?",
    ],
    "q225": [
        "Sadr Pakistan kin halat mein assembly tahleel kar sakte hain?",
        "Sadar Pakistan kin halat main assembly tahleel kar sakte hain?",
        "Sadr Pakistan kin haalat mein assembly tahleel kar sakte hain?",
    ],
    "q102": [
        "",
        "Karachi Stock Exchange 100 index kab sab se ziyada satah par pohancha tha?",
        "Karachi Stock Exchange 100 index kab sab se zyada satah par pohancha tha?",
    ],
    "q042": ["", "", "Jami'a Peshawar ke Law College ki buniyad kis saal mein rakhi gayee?"],
}

if __name__ == "__main__":
    path = ROOT / "review_sample.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    for r in rows:
        ok, note = NOTES.get(r["id"], ("yes", ""))
        r["natural_ok"] = ok
        r["notes"] = ("AI review (not a native speaker): " + note) if note else ""
        fixes = FIXES.get(r["id"], ["", "", ""])
        for k, f in enumerate(fixes, 1):
            r[f"fixed_roman_{k}"] = f
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    yes = sum(r["natural_ok"] == "yes" for r in rows)
    print(f"{yes} yes, {len(rows) - yes} no of {len(rows)}")
