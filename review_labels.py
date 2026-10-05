# AI review of the 50-question sample in review_sample.csv. Written by Claude, who is NOT a native
# Urdu speaker, so this is a second-pass sanity check and not the native-speaker review the
# benchmark ultimately needs. Codes: DRIFT = a Roman Urdu variant changes words, not just spelling;
# LEAK = the question points at "this passage" or "you" and cannot stand alone; AMBIG = ambiguous
# entity, merged question or doubtful content; SPELL = a spelling or grammar slip nobody would type.
import csv
from pathlib import Path

ROOT = Path(__file__).parent

NOTES = {
    "q179": ("no", "DRIFT: variant 2 swaps Hukoomat for Government"),
    "q005": ("no", "DRIFT: variants 2 and 3 reword (Met Office, headquarters, head office)"),
    "q191": ("no", "AMBIG: odd yes/no question. SPELL: zimmi should be zimni (fixed)"),
    "q140": ("no", "DRIFT: variant 2 swaps ilaqay for jagahen"),
    "q235": ("no", "DRIFT: kin halaat vs kis surat"),
    "q116": ("no", "DRIFT/SPELL: variant 2 uses 1D and swaps words"),
    "q285": ("no", "DRIFT: variant 3 swaps jareeda/mudeer for magazine/editor"),
    "q225": ("no", "DRIFT/SPELL: variant 1 hngami, variant 3 all caps and reworded"),
    "q188": ("no", "DRIFT: ibtida / aghaz / shuruaat are different words"),
    "q173": ("no", "AMBIG: two questions merged, favourite dishes looks invented. Suggest drop"),
    "q107": ("no", "SPELL: variant 2 grammar slip (ka qanooni nizam)"),
    "q030": ("no", "AMBIG: Murad Khan could be several people. Suggest drop"),
    "q295": ("no", "DRIFT: variants 2 and 3 reword the whole sentence"),
    "q262": ("no", "AMBIG/DRIFT: odd question, variants swap words. Suggest drop"),
    "q172": ("no", "SPELL: variant 2 khila gaya should be khela gaya (fixed)"),
    "q207": ("no", "DRIFT: mansube / plan / project"),
    "q260": ("no", "DRIFT: variant 2 asks kab tha instead of the date"),
    "q216": ("no", "AMBIG: doubtful entity. SPELL: variant 3 accent mark"),
    "q227": ("no", "DRIFT: variants 2 and 3 reword the end of the sentence"),
    "q057": ("no", "SPELL: variant 3 grammar slip (ka qaim ka faisla)"),
    "q069": ("no", "DRIFT: variants 2 and 3 reword (ODI, muqable mein, against)"),
    "q276": ("no", "SPELL: shaay / shay for shaya (fixed)"),
    "q087": ("no", "DRIFT: variant 3 swaps behas for baat"),
    "q259": ("no", "LEAK: Question prefix copied from the passage. DRIFT: variant 3 drops zero"),
    "q240": ("no", "DRIFT: variant 2 swaps kileedi kirdar for key role"),
    "q244": ("no", "LEAK: refers to this passage (is mazmoon mein)"),
    "q196": ("no", "LEAK: speaks to you (aap), cannot stand alone"),
    "q277": ("no", "SPELL: variants 2 and 3 (hwi, taamer-e-no, kub huee)"),
}
FIXES = {
    "q191": [
        "Kya 12 February 2018 ko Jahangir Tareen ki na-ahli ke baad halqa NA-154 ki zimni intikhabat mein teesri position hasil ki gayi?",
        "Kya 12 Feb 2018 ko Jahangir Tareen ki naa-ahli ke baad halqa NA-154 ki zimni election mein teesri position hasil ki gayi?",
        "",
    ],
    "q225": [
        "Sri Lanka ki cricket team ke dorah Pakistan ke doran kis waqia ke baad unhe hangami tor par wapas bheja gaya?",
        "",
        "Srilanka ki cricket team ko Pakistan ke doran kis waqia ke baad foran wapas bhej diya gaya?",
    ],
    "q107": [
        "",
        "Mughalia saltanat ke qanooni nizam mein kis shakhsiyat ka kirdar Islami qanoon ke idaron ke qiyam mein aham tha?",
        "",
    ],
    "q172": ["", "Pakistan ka pehla test match kis shahar main khela gaya?", ""],
    "q057": [
        "",
        "",
        "Minaar Pakistan ke qayam ka faisla kis jagah ke tareekhi waqiya ki yaad me kiya gaya tha?",
    ],
    "q276": [
        "Dr. Mubeen Akhtar ki kitab 'Muslamanon ke liye Jinsi Taleem' Pakistan mein kab shaya hui?",
        "Dr. Mubeen Akhtar ki kitab 'Muslmanon ke liye Jinsi Taleem' Pakistan mein kab shaaya hui?",
        "Dr. Mubeen Akhtar ki kitab 'Muslamanon k liye Jinsi Taleem' Pakistan mein kab shae hui?",
    ],
    "q277": [
        "",
        "Miyan Naseer Mohammad ki masjid ki taamir-e-nau kab hui?",
        "Mian Nasir Mohammad ki masjid ki taamir-e-nau kab hui?",
    ],
    "q216": ["", "", "Ayasho khandan ki taqat ka manba kya tha?"],
    "q179": [
        "",
        "Shahi Guzargah Project kis saal Hukumat Punjab aur Agha Khan Trust for Culture ke zariye shuru kia gaya?",
        "",
    ],
    "q285": ["", "", "Zaviya jareeda ke mudeer kaun hain?"],
    "q005": [
        "",
        "Pakistan Meteorological Department (PMD) ka sadar daftar kahan hai?",
        "Pakistan Meteorological Department ka sadar daftar kahan mojood hai?",
    ],
    "q188": [
        "",
        "Pakistan Army Aviation Corps ki ibteda kab hui?",
        "Pakistan Army Aviation Corps ki ibtidaa kab hui?",
    ],
    "q235": [
        "",
        "Sadar Pakistan kin halat mein assembly tahleel kar sakte hain?",
        "Sadar Pakistan kin haalaat mein assembly tahleel kar sakte hain?",
    ],
    "q140": ["", 'Taimur nay Iran par "Yorish Sah Saala" k doran kon se ilaqay fatah kiye?', ""],
    "q116": [
        "",
        "tisray ODI ke dauran Hashim Amla aur AB de Villiers ne kis record ke liye nai partnership qaim ki?",
        "",
    ],
    "q295": [
        "",
        "Sharif khandan ke kis fard ne teen baar Pakistan ke Wazir e Azam ki haisiyat se khidmat anjam di?",
        "Sharif khandan ke kis fard ne teen martaba Pakistan ke Wazir e Azam ki haisiyat se khidmat anjam di?",
    ],
    "q227": [
        "",
        "Razak Dawood ne kis Pakistani sadar ke dor mein mashir tijarat ki haisiyat se khidmat anjam di?",
        "Razak Dawood ne kis Pakistani sadar ke daur mein mashir tijarat ki haisiyat se khidmat anjam deen?",
    ],
    "q069": [
        "",
        "Bangladesh ke khilaf Pakistan qaumi khawateen cricket team ne March 2014 mein kitne One Day matches jeete the?",
        "Bangladesh ke khilaf Pakistan khawateen cricket team ne March 2014 mein kitnay One Day matches jeetay thay?",
    ],
    "q260": ["", "Pakistan Super League 2017 k final ki tarikh kia thi?", ""],
    "q240": [
        "",
        "Pakistan k pehle muntakhib wazeer-e-azam Zulfiqar Ali Bhutto ne kis ohde ke qiyam mein kileedi kirdar ada kia?",
        "",
    ],
    "q207": [
        "",
        "Malala Fund kay 2017 kay mansube mein kon si do mulk shamil hain?",
        "Malala Fund k 2017 k mansoobay mein konsi do mulk shaamil hain?",
    ],
    "q087": [
        "",
        "",
        "Ram Nath Sharma ki kitab 'Hindustan Mein Tareekh Taleem Ki Iqtibasat Aur Matn Ki Talaash' mein Bentink aur taleem par kya bahes ki gayi hai?",
    ],
    "q259": [
        "Dr Amjad Saqib ne kis foundation ki buniyad rakhi jo sifar sharah sood par microfinance credit program chalati hai?",
        "Dr Amjad Saqib ne kis faundeshan ki buniyad rakhi jo zero sharah sood par microfinance credit program chalaati hai?",
        "Dr Amjad Saqib ne kis foundation ki buniyad rakhi jo sifar sharah sood par microfinance credit program chalati hai?",
    ],
}
YES_NOTES = {"q179": None}

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
