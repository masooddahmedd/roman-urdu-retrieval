# Builds the passage corpus from Urdu and English Wikipedia articles on Pakistani topics. We search
# a fixed list of seed topics, pull plain-text extracts, and cut them into passages of roughly
# 60 to 170 words. Only passage ids, titles and page ids are meant to be redistributed (Wikipedia
# text is CC BY-SA), the passages themselves are rebuilt by running this script.
import json
import random
import re
import time
from pathlib import Path

import httpx

OUT = Path(__file__).parent / "data" / "passages.jsonl"
HEADERS = {
    "User-Agent": (
        "roman-urdu-retrieval/0.1 "
        "(https://github.com/masooddahmedd/roman-urdu-retrieval; research benchmark) httpx"
    )
}
SEED = 7
PASSAGES_PER_LANG = 2000
MAX_PER_ARTICLE = 4
MIN_WORDS, MAX_WORDS = 60, 170

UR_SEEDS = [
    "لاہور",
    "کراچی",
    "اسلام آباد",
    "پشاور",
    "کوئٹہ",
    "ملتان",
    "فیصل آباد",
    "راولپنڈی",
    "سیالکوٹ",
    "حیدرآباد سندھ",
    "سندھ",
    "پنجاب پاکستان",
    "بلوچستان",
    "خیبر پختونخوا",
    "گلگت بلتستان",
    "دریائے سندھ",
    "کے ٹو",
    "نانگا پربت",
    "پاکستان کرکٹ ٹیم",
    "تحریک پاکستان",
    "قائد اعظم",
    "علامہ اقبال",
    "فیض احمد فیض",
    "پاکستانی کھانے",
    "پاکستانی فلم",
    "بادشاہی مسجد",
    "موہنجو داڑو",
    "ہڑپہ",
    "معیشت پاکستان",
    "سیاست پاکستان",
    "تعلیم پاکستان",
    "زراعت پاکستان",
    "پاکستان کا موسم",
    "پاکستان ہاکی",
    "پاکستانی ڈراما",
    "اردو ادب",
    "اردو شاعری",
    "مغلیہ سلطنت",
    "لاہور قلعہ",
    "پاکستان کی یونیورسٹیاں",
    "پاکستان ریلوے",
    "پاکستان کے ڈیم",
    "پاکستان کے صوبے",
    "پاکستان کی زبانیں",
    "پاکستانی موسیقی",
    "عبدالستار ایدھی",
    "ملالہ یوسفزئی",
    "پاکستان کی تاریخ",
    "پاکستان کا آئین",
    "پاکستان کی فوج",
    "پاکستانی روپیہ",
    "کراچی بندرگاہ",
    "تھر",
    "سوات",
    "ہنزہ",
    "ٹیکسلا",
    "مینار پاکستان",
    "پاکستان سپر لیگ",
    "عمران خان",
    "نواز شریف",
    "ذوالفقار علی بھٹو",
]
EN_SEEDS = [
    "Lahore",
    "Karachi",
    "Islamabad",
    "Peshawar",
    "Quetta",
    "Multan",
    "Faisalabad",
    "Rawalpindi",
    "Sialkot",
    "Hyderabad Sindh",
    "Sindh",
    "Punjab Pakistan",
    "Balochistan",
    "Khyber Pakhtunkhwa",
    "Gilgit-Baltistan",
    "Indus River",
    "K2 mountain",
    "Nanga Parbat",
    "Pakistan national cricket team",
    "Pakistan Movement",
    "Muhammad Ali Jinnah",
    "Muhammad Iqbal poet",
    "Faiz Ahmed Faiz",
    "Pakistani cuisine",
    "Cinema of Pakistan",
    "Badshahi Mosque",
    "Mohenjo-daro",
    "Harappa",
    "Economy of Pakistan",
    "Politics of Pakistan",
    "Education in Pakistan",
    "Agriculture in Pakistan",
    "Climate of Pakistan",
    "Pakistan national field hockey team",
    "Pakistani drama",
    "Urdu literature",
    "Mughal Empire",
    "Lahore Fort",
    "Universities in Pakistan",
    "Pakistan Railways",
    "Dams in Pakistan",
    "Provinces of Pakistan",
    "Languages of Pakistan",
    "Music of Pakistan",
    "Abdul Sattar Edhi",
    "Malala Yousafzai",
    "History of Pakistan",
    "Constitution of Pakistan",
    "Pakistan Army",
    "Pakistani rupee",
    "Port of Karachi",
    "Thar Desert",
    "Swat Valley",
    "Hunza Valley",
    "Taxila",
    "Minar-e-Pakistan",
    "Pakistan Super League",
    "Imran Khan",
    "Nawaz Sharif",
    "Zulfikar Ali Bhutto",
]


def api(client: httpx.Client, lang: str, params: dict) -> dict:
    url = f"https://{lang}.wikipedia.org/w/api.php"
    for attempt in range(4):
        try:
            r = client.get(url, params={**params, "format": "json"}, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (400, 403, 404):
                raise  # a policy or request error will not fix itself, do not retry for minutes
            time.sleep(2 * (attempt + 1))
        except httpx.HTTPError:
            time.sleep(2 * (attempt + 1))
    return {}


def search_titles(client, lang: str, seeds: list[str]) -> list[dict]:
    seen, found = set(), []
    for seed in seeds:
        data = api(
            client,
            lang,
            {
                "action": "query",
                "list": "search",
                "srsearch": seed,
                "srlimit": 25,
                "srnamespace": 0,
            },
        )
        for hit in data.get("query", {}).get("search", []):
            if hit["pageid"] not in seen and hit.get("wordcount", 0) >= 400:
                seen.add(hit["pageid"])
                found.append({"pageid": hit["pageid"], "title": hit["title"]})
    return found


def fetch_text(client, lang: str, pageid: int) -> str:
    data = api(
        client,
        lang,
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "exsectionformat": "plain",
            "pageids": pageid,
        },
    )
    return data.get("query", {}).get("pages", {}).get(str(pageid), {}).get("extract", "") or ""


def to_passages(text: str) -> list[str]:
    # Paragraphs are merged until they reach the minimum length, and a paragraph that is already
    # too long is cut at sentence ends. Headings and very short fragments are dropped.
    paragraphs = [p.strip() for p in re.split(r"\n+", text) if len(p.split()) >= 8]
    passages, buf = [], []
    for p in paragraphs:
        buf.append(p)
        words = " ".join(buf).split()
        if len(words) >= MIN_WORDS:
            if len(words) > MAX_WORDS:
                words = words[:MAX_WORDS]
            passages.append(" ".join(words))
            buf = []
    return passages


def build(client, lang: str, seeds: list[str], rng: random.Random) -> list[dict]:
    articles = search_titles(client, lang, seeds)
    rng.shuffle(articles)
    out = []
    for art in articles:
        text = fetch_text(client, lang, art["pageid"])
        chunks = to_passages(text)
        rng.shuffle(chunks)
        for i, chunk in enumerate(chunks[:MAX_PER_ARTICLE]):
            out.append(
                {
                    "id": f"{lang}-{art['pageid']}-{i}",
                    "lang": lang,
                    "title": art["title"],
                    "pageid": art["pageid"],
                    "text": chunk,
                }
            )
        if len(out) >= PASSAGES_PER_LANG:
            break
    return out[:PASSAGES_PER_LANG]


if __name__ == "__main__":
    rng = random.Random(SEED)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client() as client:
        passages = build(client, "ur", UR_SEEDS, rng) + build(client, "en", EN_SEEDS, rng)
    with OUT.open("w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    counts = {lang: sum(p["lang"] == lang for p in passages) for lang in ("ur", "en")}
    print(counts)
