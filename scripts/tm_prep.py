import pandas as pd, re, collections, pickle
df=pd.read_pickle('df.pkl')
df['ctx']=(df.left.fillna('')+' '+df.right.fillna('')).str.replace(r'</?s>',' ',regex=True)
# near-duplicate removal: same 12-word window around the keyword
key=(df.left.fillna('').str.split().str[-6:].str.join(' ')+'|'+df.right.fillna('').str.split().str[:6].str.join(' '))
df=df[~key.duplicated()].copy()
STOP=set("""של את על עם זה זו זאת הוא היא הם הן אני אנחנו אתה את אתם לא כן גם או אבל כי אם כל יש אין היה היתה היו יהיה להיות
מה מי איך למה כמו רק עוד כבר מאוד יותר פחות אחד אחת שני שתי בין אל עד אחרי לפני תחת מתוך בתוך בלי ללא כך כאן שם אז עכשיו
אותו אותה אותם שלו שלה שלהם שלי שלנו שלך לו לה להם לי לנו לך בו בה בהם בי בנו ממנו ממנה מהם אשר כאשר כדי בגלל לכן אולם אך
הזה הזאת האלה אלה אלו כאילו בעצם אפילו עצמו עצמה עצמם כלל כמה הרבה מעט די רבה רבים רבות מן כפי אף שוב תמיד לעולם הייתה
דרך פי וכו וגם וכן ולא ואת וזה שהוא שהיא שלא שזה שאין שיש כשהוא היום שנים שנה זמן ו ה ב ל מ ש כ את' יותר' כמיהה
דבר דברים אולי משהו פעם פעמים רוצה רוצים רצה בכל טוב טובה אותי אותך אותנו אותם רגע יכול יכולה יכולים אפשר הזו הזה יום ימים הכל
קשר בלתי אלא ידי וה באופן אינה אינו אינם חלק אנו מכן אחר אחרת אחרים שבו שבה זהו ועל בשנת המשך הרי עדיין ממש לגבי אצל כיצד
ככל כמובן שוב בכך בזה לכך מכל נראה נראית צריך צריכה צריכים לעשות עושה עושים אמר אומר אמרה אומרת כאן לפעמים אחד אחת שלושה
גדול גדולה חדש חדשה ראשון ראשונה אחרון אחרונה אותה כולם כולנו לכל מהם שהם שהיה שהייתה שבין שבו שכן כאשר כמעט בלבד מעבר בעיקר""".split())
tok=lambda s:[w for w in re.findall(r'[א-ת]{2,}',re.sub(r'[֑-ׇ]','',s))]
docs=[tok(s) for s in df.ctx]
freq=collections.Counter(w for d in docs for w in d)
PREF="והבלמשכ"
def norm(w):
    for _ in range(2):
        if len(w)>3 and w[0] in PREF and freq.get(w[1:],0)>=max(5,freq[w]): w=w[1:]
        else: break
    return w
cmap={w:norm(w) for w in freq}
KM=re.compile(r'^כמיה')
docs=[[cmap[w] for w in d] for d in docs]
docs=[[w for w in d if w not in STOP and not KM.match(w) and len(w)>1] for d in docs]
df['tokens']=[' '.join(d) for d in docs]
df=df[df.tokens.str.split().str.len()>=8]
df[['example_id','source','tokens','t','dom']].to_pickle('tm/docs.pkl')
print(len(df)); print(df.tokens.iloc[0][:200])
