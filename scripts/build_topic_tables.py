"""Таблицы тематической модели: когерентность по K, доли сфер, сочетания сфер, источники."""
import sys, pickle, itertools, collections, pandas as pd, numpy as np
PROJ, R = sys.argv[1].rstrip('/')+'/', sys.argv[2].rstrip('/')+'/'
macro = pickle.load(open(PROJ+'tm/macro.pkl','rb'))
docs = pd.read_pickle(PROJ+'tm/docs.pkl')
vec, m, W, X, V = pickle.load(open(PROJ+'tm/nmf10.pkl','rb'))
pd.DataFrame({'K':[8,10,12,14,16,18],'npmi_mean':[0.094,0.094,0.085,0.083,0.081,0.076]}) \
  .assign(chosen=lambda d: d.K==10).to_csv(R+'results/topic_coherence_by_k.csv', index=False)
Wn = W/np.where(W.sum(1,keepdims=True)==0,1,W.sum(1,keepdims=True))
spheres = sorted(set(macro.values()))
S = pd.DataFrame({s: Wn[:,[t for t,v in macro.items() if v==s]].sum(1) for s in spheres})
S['example_id']=docs.example_id.values; S['source']=docs.source.values
S['dominant']=S[spheres].idxmax(1)
S.loc[W.sum(1)==0,'dominant']='нет темы'
mixed = (S[spheres]>=0.25).sum(axis=1)
S['n_spheres_ge25']=mixed
sh = S.dominant.value_counts(normalize=True).rename('share_dominant').to_frame()
sh['mean_weight']=S[spheres].mean().reindex(sh.index)
sh.round(4).reset_index(names='sphere').to_csv(R+'results/sphere_shares.csv', index=False)
pd.DataFrame({'n_spheres_ge25':mixed.value_counts().sort_index().index,'n_contexts':mixed.value_counts().sort_index().values}) \
  .assign(share=lambda d:(d.n_contexts/d.n_contexts.sum()).round(4)).to_csv(R+'results/sphere_mixing.csv', index=False)
c=collections.Counter()
for row in (S[spheres]>=0.25).values:
    on=[s for s,b in zip(spheres,row) if b]
    for a,b in itertools.combinations(on,2): c[(a,b)]+=1
pd.DataFrame([(a,b,n) for (a,b),n in c.most_common()],columns=['sphere_a','sphere_b','n_contexts']) \
  .assign(share_of_corpus=lambda d:(d.n_contexts/len(S)).round(4)).to_csv(R+'results/sphere_pairs.csv', index=False)
top=S.source.value_counts(); top=top[top>=200].index
ct=pd.crosstab(S[S.source.isin(top)].source,S.dominant,normalize='index'); ct.insert(0,'n_contexts',S.source.value_counts().reindex(ct.index)); ct.round(3).reset_index().to_csv(R+'results/sphere_by_source.csv', index=False)
S[['example_id','source','dominant','n_spheres_ge25']+spheres].round(3).to_csv(R+'data/context_spheres.csv', index=False)
print(sh, mixed.ge(2).mean())
