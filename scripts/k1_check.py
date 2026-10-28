import json, re, pymorphy3
m = pymorphy3.MorphAnalyzer()
res = json.load(open('results.json'))

# --- allowed vocab: primes (+allolexes) + grammatical links/pronouns ---
RU = set("""я ты кто-то что-то вещь люди тело род вид часть этот тот же самый другой один два некоторый весь много мало
хороший плохой большой маленький думать знать хотеть чувствовать видеть слышать сказать говорить слово правда
делать происходить случиться двигаться быть есть жить умереть когда время сейчас до после долго недолго некоторое момент
где место здесь над под далеко близко сторона внутри касаться не может мочь потому что если очень больше как так
человек кто что это у в на о об с и а чтобы такой он она они его её их ему ей им меня мне мой свой себя тот это
иногда некоторое когда-то тогда там нет""".split())
EN = set("""i you someone something thing people body kind part this the same other else one two some all much many little few
good bad big small think know want feel see hear say word words true do happen move be is are was were am there mine
live die when time now before after long short moment where place here above below far near side inside touch not no
maybe can because if very more like as way person a an it its they them their he she his her me my of to for at in on
about with that what so how these those someone's times have has thing's""".split())
HE = set("""אני אתה את מישהו משהו אנשים גוף סוג חלק זה הזה הזאת זאת אותו הדבר אחר אחד שניים כל הכול הרבה מעט טוב רע גדול קטן
חשב חושב חושבים ידע יודע רצה רוצה רוצים הרגיש מרגיש מרגישים ראה רואה שמע שומע אמר אומר מילים נכון עשה עושה קרה קורה זז נע
נגע נוגע חי מת מתי זמן עכשיו לפני אחרי קצר למשך מה רגע איפה מקום פה כאן מעל מתחת רחוק קרוב צד בתוך לא אולי יכול בגלל אם מאוד יותר כמו
יש שלי הוא היא הם להיות נמצא כך כמה ש של לו לה""".split())
PREF = "השוכבלמ"

def tokens(s):
    return re.findall(r"[A-Za-z']+|[֐-׿]+|[а-яёА-ЯЁ-]+|[一-鿿]+", s)

def bad_ru(t):
    w = t.lower().strip('-')
    if not w: return False
    if w in RU: return False
    return m.parse(w)[0].normal_form not in RU
def bad_en(t):
    w = t.lower().strip("'")
    for cand in [w, w[:-1] if w.endswith('s') else w, w[:-2] if w.endswith('es') else w, w[:-1] if w.endswith('d') else w]:
        if cand in EN: return False
    return True
def bad_he(t):
    if t in HE: return False
    for i in range(1,3):
        if t[i:] in HE and all(c in PREF for c in t[:i]): return False
    return True

def check(lang, lines):
    out = []
    for line in lines:
        for t in tokens(line):
            if re.match(r"[一-鿿]", t): out.append(t); continue
            if re.match(r"[A-Za-z]", t):
                if lang=='en': 
                    if bad_en(t): out.append(t)
                else: out.append(t)          # латиница в ru/he = чужой язык
            elif re.match(r"[֐-׿]", t):
                if lang!='he' or bad_he(t): out.append(t)
            else:
                if lang!='ru' or bad_ru(t): out.append(t)
    return out

rows=[]
for r in res:
    ex = (r['parsed'] or {}).get('explication', []) if r['raw'] else []
    bad = check(r['lang'], ex) if ex else None
    rows.append(dict(model=r['model'],lang=r['lang'],iter=r['iter'],n_lines=len(ex),bad=bad))
json.dump(rows, open('k1/auto.json','w'), ensure_ascii=False, indent=1)
for x in rows:
    b = x['bad']
    print(x['model'],x['lang'],x['iter'], 'ОТКАЗ' if b is None else f"{len(set(b))} | {sorted(set(b))[:25]}")
