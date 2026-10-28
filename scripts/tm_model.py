import pandas as pd, numpy as np, sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
d=pd.read_pickle('tm/docs.pkl')
vec=TfidfVectorizer(min_df=15,max_df=0.15,token_pattern=r'\S+',sublinear_tf=True)
X=vec.fit_transform(d.tokens); V=np.array(vec.get_feature_names_out()); print(X.shape)
def coherence(H,k=10):
    # NPMI-ish via document co-occurrence on binary matrix
    B=(X>0).astype(np.float32).tocsc(); N=B.shape[0]; sc=[]
    for t in H:
        top=np.argsort(-t)[:k]; s=[]
        for i in range(k):
            for j in range(i+1,k):
                a,b=top[i],top[j]; pa=B[:,a].sum()/N; pb=B[:,b].sum()/N; pab=B[:,a].multiply(B[:,b]).sum()/N
                if pab>0: s.append(np.log(pab/(pa*pb))/-np.log(pab))
                else: s.append(-1)
        sc.append(np.mean(s))
    return np.mean(sc)
for K in [int(k) for k in sys.argv[1:]]:
    m=NMF(K,init='nndsvda',random_state=0,max_iter=400); W=m.fit_transform(X)
    print('K',K,'NPMI',round(coherence(m.components_),3))
