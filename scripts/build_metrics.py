"""Собирает все таблицы data/ и results/ репозитория из базы проекта и экспертных оценок.
Запуск из папки проекта: python scripts/build_metrics.py <PROJ> <REPO>"""
import sys, json, re, sqlite3, itertools, collections, shutil
import pandas as pd, numpy as np
from scipy.stats import kruskal, mannwhitneyu, spearmanr

PROJ, R = sys.argv[1].rstrip('/') + '/', sys.argv[2].rstrip('/') + '/'
con = sqlite3.connect(PROJ + 'out/kmiha_project.db')
runs = pd.read_sql('select * from runs', con)
sc = pd.read_pickle(PROJ + 'scores_expert.pkl')

# ---------------- data/ ----------------
pd.read_csv(PROJ + 'contexts_24.csv', encoding='utf-8-sig').to_csv(R + 'data/contexts_24.csv', index=False)
pd.read_sql('select * from primes', con).to_csv(R + 'data/primes_ru_en_he.csv', index=False)
pd.read_sql('select * from models', con).to_csv(R + 'data/models.csv', index=False)
pd.read_sql('select * from rubric', con).to_csv(R + 'data/rubric.csv', index=False)
r = runs[['code', 'model', 'lang', 'iter', 'status', 'n_lines', 'spheres_json', 'explication', 'raw_response']].copy()
r['spheres'] = r.spheres_json.apply(lambda s: '; '.join(json.loads(s)) if isinstance(s, str) else '')
r['explication'] = r.explication.fillna('')
r = r.drop(columns='spheres_json').sort_values(['model', 'lang', 'iter'])
r.to_csv(R + 'data/responses_long.csv', index=False)
shutil.copy(PROJ + 'results.json', R + 'data/model_responses.json')

w = sc[['code', 'model', 'lang', 'iter', 'status', 'K1', 'K2', 'K3', 'K4', 'K5', 'total']].sort_values(['model', 'lang', 'iter'])
w.to_csv(R + 'data/scores_wide.csv', index=False)
sl = w.melt(id_vars=['code', 'model', 'lang', 'iter', 'status'], value_vars=['K1', 'K2', 'K3', 'K4', 'K5'],
            var_name='criterion', value_name='score')
k1r = dict(con.execute('select code,k1_reason from scores'))
sl['rater'] = np.where(sl.criterion == 'K1', 'algorithm+manual', 'expert')
sl['note'] = np.where(sl.criterion == 'K1', sl.code.map(k1r), '')
sl.to_csv(R + 'data/scores_long.csv', index=False)

# ---------------- results/ ----------------
def agg(g):
    t = g['total']
    return pd.Series({'n': len(t), 'mean': t.mean(), 'sd': t.std(), 'cv': t.std() / t.mean() if t.mean() else np.nan,
                      'min': t.min(), 'median': t.median(), 'max': t.max(),
                      'refusals': (g.status == 'отказ').sum(), 'broken': (g.status != 'ok').sum()})
sc.groupby(['model', 'lang']).apply(agg).round(3).reset_index().to_csv(R + 'results/summary_model_lang.csv', index=False)
sc.groupby('lang').apply(agg).round(3).reset_index().to_csv(R + 'results/by_language.csv', index=False)
sc.groupby('model').apply(agg).round(3).reset_index().to_csv(R + 'results/by_model.csv', index=False)
sl.groupby(['model', 'lang', 'criterion'])['score'].agg(['mean', 'std', 'min', 'max']).round(3).reset_index() \
  .to_csv(R + 'results/by_criterion.csv', index=False)
sl.groupby('criterion')['score'].agg(['mean', 'std']).round(3).reset_index().to_csv(R + 'results/criterion_overall.csv', index=False)

r.groupby(['model', 'lang', 'status']).size().reset_index(name='n').to_csv(R + 'results/failures.csv', index=False)
pd.DataFrame([['Dots3', 'he', 5], ['Dots3', 'en', 3], ['Dots3', 'ru', 2], ['Nemotron3', 'ru', 1]],
             columns=['model', 'lang', 'empty_answers']).assign(run='пробный прогон: max_tokens=8000, без повтора пустого ответа') \
  .to_csv(R + 'results/pilot_empty_answers.csv', index=False)

# lexical consistency across the 5 iterations (mean pairwise Jaccard)
tok = lambda s: set(re.findall(r'\w+', (s or '').lower()))
def jac(A): return [len(a & b) / len(a | b) for a, b in itertools.combinations(A, 2) if a | b]
rows = []
for (m, l), g in r.groupby(['model', 'lang']):
    ex = [tok(e) for e in g.explication if e]
    sp = [set(x.strip().lower() for x in s.split(';') if x.strip()) for s in g.spheres if s]
    je, js = jac(ex), jac(sp)
    rows.append([m, l, len(ex), np.mean(je) if je else np.nan, np.mean(js) if js else np.nan])
pd.DataFrame(rows, columns=['model', 'lang', 'answers', 'explication_jaccard_mean', 'spheres_jaccard_mean']).round(3) \
  .to_csv(R + 'results/consistency_lexical.csv', index=False)

r['words'] = r.explication.str.split().str.len()
r['n_spheres'] = r.spheres.apply(lambda s: len([x for x in s.split(';') if x.strip()]))
ln = r.groupby(['model', 'lang'])[['n_lines', 'words', 'n_spheres']].agg(['mean', 'std', 'min', 'max']).round(2)
ln.columns = ['_'.join(c) for c in ln.columns]; ln.reset_index().to_csv(R + 'results/explication_length.csv', index=False)

auto = json.load(open(PROJ + 'k1/auto.json'))
ALLOW = {"'", 'sometimes', 'from', 'из-за', 'для', 'לחשוב'}
c = collections.Counter()
for a in auto:
    for wd in {x.lower().strip("'") for x in (a['bad'] or [])}:
        if wd and wd not in ALLOW: c[(a['model'], a['lang'], wd)] += 1
pd.DataFrame([(m, l, wd, n) for (m, l, wd), n in c.items()], columns=['model', 'lang', 'word', 'answers_with_word']) \
  .sort_values(['answers_with_word', 'model'], ascending=[False, True]).to_csv(R + 'results/k1_out_of_list_words.csv', index=False)

def drift(row):
    t = row.explication
    n = {'hebrew': len(re.findall(r'[֐-׿]', t)), 'cyrillic': len(re.findall(r'[а-яёА-ЯЁ]', t)),
         'latin': len(re.findall(r'[A-Za-z]', t)), 'chinese': len(re.findall(r'[一-鿿]', t))}
    tot = sum(n.values())
    out = {k: (v / tot if tot else np.nan) for k, v in n.items()}
    out['share_in_prompt_script'] = (n[{'he': 'hebrew', 'ru': 'cyrillic', 'en': 'latin'}[row.lang]] / tot) if tot else np.nan
    return pd.Series(out)
pd.concat([r[['code', 'model', 'lang', 'iter', 'status']], r.apply(drift, axis=1).round(3)], axis=1) \
  .to_csv(R + 'results/script_drift.csv', index=False)

pat = {'national': r'национ|этнич|родин|patriot|nation|ethnic|לאומ|מולדת|לאום|אומ|territor',
       'religious': r'религ|духов|relig|spirit|דת|רוח|אמונה|Бог',
       'interpersonal': r'межлич|личн|отношен|relationship|interpersonal|יחס|близост|bonding|union',
       'erotic': r'эрот|сексу|интим|физическ|физиолог|телесн|romantic|intimate|sexual|קרבה פיזית|צורך גופני',
       'psych': r'психол|душев|терап|благопол|экзистенц|psycholog|therapy|well-being|existential|נפש|psyche|שלווה|emotional|эмоцион',
       'family_child': r'ребён|родител|offspring|parental|procreat|ילדים|משפחה|family'}
cov = r[r.status != 'отказ'].copy()
for k, p in pat.items(): cov[k] = cov.spheres.str.contains(p, case=False, regex=True)
cov.groupby(['model', 'lang'])[list(pat)].sum().reset_index().to_csv(R + 'results/sphere_coverage_by_model_lang.csv', index=False)

st = []
k = kruskal(*[g.total for _, g in sc.groupby('lang')]); st.append(['Kruskal–Wallis: итог по языкам (обе модели)', k.statistic, k.pvalue, 30])
for m, g in sc.groupby('model'):
    k = kruskal(*[x.total for _, x in g.groupby('lang')]); st.append([f'Kruskal–Wallis: итог по языкам, {m}', k.statistic, k.pvalue, 15])
u = mannwhitneyu(sc[sc.model == 'Nemotron3'].total, sc[sc.model == 'Dots3'].total)
st.append(['Mann–Whitney: Nemotron3 vs Dots3 (все языки)', u.statistic, u.pvalue, 30])
for l in ['ru', 'en', 'he']:
    u = mannwhitneyu(sc[(sc.model == 'Nemotron3') & (sc.lang == l)].total, sc[(sc.model == 'Dots3') & (sc.lang == l)].total)
    st.append([f'Mann–Whitney: Nemotron3 vs Dots3, {l}', u.statistic, u.pvalue, 10])
words = r.set_index('code').loc[sc.code, 'words'].values
s = spearmanr(words, sc.total.values); st.append(['Spearman: длина толкования (слова) vs итог', s.statistic, s.pvalue, 30])
pd.DataFrame(st, columns=['test', 'statistic', 'p_value', 'n']).round(4).to_csv(R + 'results/stat_tests.csv', index=False)

pd.read_csv(PROJ + 'tm/topics.csv', encoding='utf-8-sig').to_csv(R + 'results/topics_nmf10.csv', index=False)
r.to_pickle(PROJ + 'r_lengths.pkl')
print('done')
