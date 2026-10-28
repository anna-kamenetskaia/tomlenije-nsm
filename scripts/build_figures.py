"""Рисунки репозитория из таблиц results/ и data/."""
import sys, pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = sys.argv[1].rstrip('/')+'/'; F = R+'figures/'
C = {'Nemotron3':'#2a78d6','Dots3':'#eb6834'}; L=['ru','en','he']; LN={'ru':'русский','en':'английский','he':'иврит'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.grid':True,'grid.color':'#e6e6e6','axes.axisbelow':True,'figure.dpi':150})
def save(n): plt.tight_layout(); plt.savefig(F+n, bbox_inches='tight'); plt.close()
w = pd.read_csv(R+'data/scores_wide.csv'); r = pd.read_csv(R+'data/responses_long.csv').fillna({'explication':''})

# 1. распределение итоговых баллов (точки по итерациям)
fig,ax=plt.subplots(figsize=(8,4.2)); x=np.arange(3)
for i,(m,off) in enumerate([('Nemotron3',-.17),('Dots3',.17)]):
    for j,l in enumerate(L):
        v=w[(w.model==m)&(w.lang==l)].total.values
        ax.scatter(np.full(len(v),j+off)+np.linspace(-.05,.05,len(v)),v,color=C[m],alpha=.75,s=40,label=m if j==0 else None)
        ax.hlines(v.mean(),j+off-.12,j+off+.12,color=C[m],lw=2.5)
ax.set_xticks(x,[LN[l] for l in L]); ax.set_ylabel('итоговый балл (0–10)'); ax.set_ylim(-.5,10.5)
ax.set_title('Итоговые баллы по итерациям (черта — среднее)',loc='left'); ax.legend(frameon=False); save('fig_scores_by_iteration.png')

# 2. длина толкования vs итог
d=w.merge(r[['code','explication']],on='code'); d['words']=d.explication.str.split().str.len()
fig,ax=plt.subplots(figsize=(7,4.2))
for m in C: s=d[d.model==m]; ax.scatter(s.words,s.total,color=C[m],s=40,alpha=.8,label=m)
ax.set_xlabel('длина толкования, слов'); ax.set_ylabel('итоговый балл'); ax.legend(frameon=False)
ax.set_title('Длина толкования и оценка (Spearman ρ = 0,71)',loc='left'); save('fig_length_vs_score.png')

# 3. согласованность (Жаккар) по итерациям
c=pd.read_csv(R+'results/consistency_lexical.csv')
fig,ax=plt.subplots(figsize=(7,4))
for i,(m,off) in enumerate([('Nemotron3',-.2),('Dots3',.2)]):
    s=c[c.model==m].set_index('lang').reindex(L)
    ax.bar(np.arange(3)+off,s.explication_jaccard_mean,.38,color=C[m],label=m)
ax.set_xticks(range(3),[LN[l] for l in L]); ax.set_ylim(0,1); ax.legend(frameon=False)
ax.set_ylabel('средний Жаккар между итерациями'); ax.set_title('Лексическая стабильность толкований',loc='left'); save('fig_consistency.png')

# 4. вариативность: CV
sm=pd.read_csv(R+'results/summary_model_lang.csv')
fig,ax=plt.subplots(figsize=(7,4))
for m,off in [('Nemotron3',-.2),('Dots3',.2)]:
    s=sm[sm.model==m].set_index('lang').reindex(L); ax.bar(np.arange(3)+off,s.cv,.38,color=C[m],label=m)
ax.set_xticks(range(3),[LN[l] for l in L]); ax.legend(frameon=False); ax.set_ylabel('коэффициент вариации (SD / среднее)')
ax.set_title('Разброс оценок между итерациями',loc='left'); save('fig_cv.png')

# 5. смешение письменностей
sd=pd.read_csv(R+'results/script_drift.csv'); sd=sd[sd.status!='отказ']
g=sd.groupby(['model','lang'])[['hebrew','cyrillic','latin','chinese']].mean()
fig,ax=plt.subplots(figsize=(8,4)); cols={'hebrew':'#5b8a3a','cyrillic':'#2a78d6','latin':'#9a9a9a','chinese':'#c2304a'}
lab=[f'{m}\n{LN[l]}' for m in ['Nemotron3','Dots3'] for l in L]; left=np.zeros(6)
for k,col in cols.items():
    v=np.array([g.loc[(m,l),k] if (m,l) in g.index else 0 for m in ['Nemotron3','Dots3'] for l in L])
    ax.barh(range(6),v,left=left,color=col,label={'hebrew':'иврит','cyrillic':'кириллица','latin':'латиница','chinese':'китайские иероглифы'}[k]); left+=v
ax.set_yticks(range(6),lab); ax.invert_yaxis(); ax.set_xlim(0,1); ax.grid(axis='y',visible=False)
ax.set_xlabel('доля символов в толковании'); ax.legend(frameon=False,ncol=4,loc='upper center',bbox_to_anchor=(.5,-.18))
ax.set_title('В какой письменности написано толкование',loc='left'); save('fig_script_drift.png')

# 6. средние по критериям
bc=pd.read_csv(R+'results/by_criterion.csv'); K=['K1','K2','K3','K4','K5']
fig,axs=plt.subplots(1,3,figsize=(11,3.6),sharey=True)
for ax,l in zip(axs,L):
    for m,off in [('Nemotron3',-.2),('Dots3',.2)]:
        s=bc[(bc.model==m)&(bc.lang==l)].set_index('criterion').reindex(K); ax.bar(np.arange(5)+off,s['mean'],.38,color=C[m],label=m)
    ax.set_xticks(range(5),K); ax.set_ylim(0,2); ax.set_title(LN[l],loc='left')
axs[0].set_ylabel('средний балл (0–2)'); axs[0].legend(frameon=False); save('fig_criteria_bars.png')

# 7. когерентность тем по K
co=pd.read_csv(R+'results/topic_coherence_by_k.csv')
fig,ax=plt.subplots(figsize=(6,3.6)); ax.plot(co.K,co.npmi_mean,'-o',color='#2a78d6')
ax.scatter([10],[co.set_index('K').npmi_mean[10]],s=120,facecolors='none',edgecolors='#eb6834',lw=2,zorder=3)
ax.set_xlabel('число тем K'); ax.set_ylabel('средняя NPMI'); ax.set_title('Выбор числа тем: когерентность NPMI',loc='left'); save('fig_topic_coherence.png')

# 8. пары сфер
p=pd.read_csv(R+'results/sphere_pairs.csv').head(10)[::-1]
fig,ax=plt.subplots(figsize=(7,4)); ax.barh(p.sphere_a+' + '+p.sphere_b,p.share_of_corpus*100,color='#2a78d6')
ax.grid(axis='y',visible=False); ax.set_xlabel('% контекстов корпуса (обе сферы ≥ 25% веса)')
ax.set_title('Какие сферы встречаются вместе',loc='left'); save('fig_sphere_pairs.png')

# 9. число сфер на контекст
mx=pd.read_csv(R+'results/sphere_mixing.csv')
fig,ax=plt.subplots(figsize=(6,3.6)); ax.bar(mx.n_spheres_ge25.astype(str),mx.share*100,color=['#9a9a9a','#9a9a9a']+['#eb6834']*(len(mx)-2))
ax.set_xlabel('сколько сфер имеют ≥ 25% веса в контексте'); ax.set_ylabel('% контекстов')
ax.set_title(f'Смешанных контекстов: {mx[mx.n_spheres_ge25>=2].share.sum()*100:.0f}%',loc='left'); save('fig_sphere_mixing.png')

# 10. сферы по источникам
sb=pd.read_csv(R+'results/sphere_by_source.csv').set_index('source').drop(columns='n_contexts')
sp=['Психология','Нация','Религия','Искусство','Любовь и эрос']; sb=sb[sp].sort_values('Психология')
pal=['#2a78d6','#eb6834','#5b8a3a','#8e5bb5','#c2304a']
fig,ax=plt.subplots(figsize=(8,5)); left=np.zeros(len(sb))
for s,col in zip(sp,pal): ax.barh(sb.index,sb[s],left=left,color=col,label=s); left+=sb[s].values
ax.set_xlim(0,1); ax.grid(axis='y',visible=False); ax.legend(frameon=False,ncol=5,loc='upper center',bbox_to_anchor=(.45,-.1),fontsize=9)
ax.set_title('Ведущая сфера по крупным источникам (≥ 200 контекстов)',loc='left'); save('fig_sphere_by_source.png')

# 11. слова вне списка примитивов
k=pd.read_csv(R+'results/k1_out_of_list_words.csv').groupby('word').answers_with_word.sum().sort_values().tail(15)
fig,ax=plt.subplots(figsize=(6,4.5)); ax.barh(k.index,k.values,color='#eb6834'); ax.grid(axis='y',visible=False)
ax.set_xlabel('в скольких ответах встречается'); ax.set_title('Частые слова вне списка примитивов (К1)',loc='left'); save('fig_k1_words.png')
print('ok')
