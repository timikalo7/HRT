import numpy as np, pandas as pd
from lib import load
import irf
piv, hl, sess, ret = load("ins")
IND = {"tech":"Systems Microchips Computing Software Technologies Networks Devices Solutions".split(),
 "health":"Sciences Genomics Labs Diagnostics Biotech Pharmaceuticals Therapeutics Biopharma Biosciences".split(),
 "consumer":"Commerce Marketplace Retail Outlets Brands Stores Goods".split(),
 "fin":"Securities Capital Investments Holdings Asset Financial Trading Partners Advisors".split(),
 "energy":"Fuels Power Resources Grid Renewables Energy Industries".split()}
w2i={w:k for k,v in IND.items() for w in v}
hl["suf"]=hl.co.map(lambda c: next((w for w in c.split()[1:] if w in w2i),"?"))
hl["ind"]=hl.suf.map(w2i)
vis=hl[hl.bar_ix<=49].drop_duplicates(["si","co"])
cs=vis.groupby(["si","ind"]).size().unstack(fill_value=0)
main=cs.idxmax(axis=1); mx=cs.max(axis=1); ties=(cs.eq(mx,axis=0)).sum(axis=1)
hl["main"]=hl.si.map(main); hl["inmain"]=hl.ind==hl.main; hl["mx"]=hl.si.map(mx); hl["tie"]=hl.si.map(ties)>1
hl["nind"]=hl.set_index(["si","ind"]).index.map(cs.stack()).fillna(0)
tm={t:i for i,t in enumerate(sorted(hl.tmpl.unique()))}; hl["tid"]=hl.tmpl.map(tm); n=len(sess)
a,k=irf.fit(piv["close"],hl,np.arange(0,n,2),len(tm))
C=piv["close"]; lc=np.log(np.c_[np.ones(n),C])
h=hl[(hl.si%2==1)&(hl.bar_ix<=84)].copy(); h["a"]=a[h.tid]; s=h.si.values; t=h.bar_ix.values
h["m"]=lc[s,t+11]-lc[s,t]
sl=lambda g: pd.Series({"n":len(g),"slope":round(np.sum(g.a*g.m)/np.sum(g.a**2),2)})
print(h.groupby(["inmain","mx"]).apply(sl).to_string())
print(h.groupby(["inmain","tie"]).apply(sl).to_string())
print(h[h.inmain].groupby("suf").apply(sl).sort_values("slope").to_string())
print(h[~h.inmain].groupby("suf").apply(sl).sort_values("slope").to_string())
# headline topic words: per template slope in-main split halves? check event-type (first word) in main
h["verb"]=h.tmpl.str.split().str[0]
print(h[h.inmain].groupby("verb").apply(sl).sort_values("slope").to_string())
h.to_parquet("/tmp/claude-0/-home-user-HRT/c4778cfc-8477-55f2-820f-bf5e776a3138/scratchpad/h10.parquet")
