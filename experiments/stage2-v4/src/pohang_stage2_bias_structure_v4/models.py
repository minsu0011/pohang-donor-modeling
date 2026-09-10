from __future__ import annotations

from dataclasses import dataclass
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pohang_main_v2.modeling import _prepare_catboost


@dataclass(frozen=True)
class StructureCandidate:
    family: str
    profile: str
    params_json: str
    @classmethod
    def make(cls,family:str,profile:str,**params):
        return cls(family,profile,json.dumps(params,sort_keys=True,ensure_ascii=False))
    @property
    def params(self): return json.loads(self.params_json)
    @property
    def id(self):
        p=self.params; tail="_".join(f"{k}-{p[k]}" for k in sorted(p)) if p else "default"
        return f"{self.family}__{self.profile}__{tail}"


def structure_candidates(profiles:list[str],quick:bool=False)->list[StructureCandidate]:
    out=[StructureCandidate.make("NO_STRUCTURE","FULL_ORTHO")]
    for p in profiles:
        for a in ((1.0,100.0) if quick else (0.1,1.0,10.0,100.0,1000.0)):
            out.append(StructureCandidate.make("RIDGE",p,alpha=a))
        if not quick:
            for a in (0.001,0.01,0.1):
                for l1 in (0.05,0.2,0.5): out.append(StructureCandidate.make("ELASTICNET",p,alpha=a,l1_ratio=l1))
            for eps in (1.2,1.35,1.5): out.append(StructureCandidate.make("HUBER",p,epsilon=eps,alpha=0.0001))
        out.append(StructureCandidate.make("CATBOOST_CONSERVATIVE",p,iterations=140 if quick else 350,depth=4,learning_rate=0.025,l2_leaf_reg=15.0,random_strength=1.0))
    return out


def full_catboost_candidates(profiles:list[str],quick:bool=False):
    it=180 if quick else 900
    return [StructureCandidate.make("CATBOOST_FULL",p,iterations=it,depth=6 if quick else 7,learning_rate=0.035,l2_leaf_reg=7.0,random_strength=0.45) for p in profiles]


def _pipeline(family,numeric,categorical,params):
    num=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())])
    cat=Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))])
    pre=ColumnTransformer([("num",num,numeric),("cat",cat,categorical)],remainder="drop",sparse_threshold=0.0)
    if family=="RIDGE": est=Ridge(alpha=float(params["alpha"]),fit_intercept=True)
    elif family=="ELASTICNET": est=ElasticNet(alpha=float(params["alpha"]),l1_ratio=float(params["l1_ratio"]),max_iter=10000,tol=1e-5)
    elif family=="HUBER": est=HuberRegressor(epsilon=float(params["epsilon"]),alpha=float(params.get("alpha",0.0001)),max_iter=1000)
    else: raise ValueError(family)
    return Pipeline([("pre",pre),("model",est)])


def fit_predict_centered(cand:StructureCandidate,train:pd.DataFrame,valid:pd.DataFrame,features:list[str],categorical:list[str],*,target_col="__centered_residual",backend="CPU",seed=4242,threads=8,gpu_ram_part=0.92,used_ram_limit="80gb"):
    if cand.family=="NO_STRUCTURE":
        return np.zeros(len(train),float),np.zeros(len(valid),float),0.0
    y=pd.to_numeric(train[target_col],errors="coerce").to_numpy(float)
    cats=[c for c in categorical if c in features]; nums=[c for c in features if c not in set(cats)]
    if cand.family in {"RIDGE","ELASTICNET","HUBER"}:
        m=_pipeline(cand.family,nums,cats,cand.params); m.fit(train[features],y)
        ptr=np.asarray(m.predict(train[features]),float); pva=np.asarray(m.predict(valid[features]),float)
    else:
        from catboost import CatBoostRegressor,Pool
        p=cand.params
        kw={"iterations":int(p["iterations"]),"depth":int(p["depth"]),"learning_rate":float(p["learning_rate"]),"l2_leaf_reg":float(p["l2_leaf_reg"]),"random_strength":float(p["random_strength"]),"loss_function":"RMSE","eval_metric":"RMSE","random_seed":int(seed),"verbose":False,"allow_writing_files":False,"task_type":backend.upper(),"thread_count":int(threads),"used_ram_limit":str(used_ram_limit)}
        if backend.upper()=="GPU": kw.update(devices="0",gpu_ram_part=float(gpu_ram_part))
        Xtr=_prepare_catboost(train,features,cats); Xv=_prepare_catboost(valid,features,cats); idx=[features.index(c) for c in cats]
        m=CatBoostRegressor(**kw); m.fit(Pool(Xtr,y,cat_features=idx),verbose=False)
        ptr=np.asarray(m.predict(Pool(Xtr,cat_features=idx)),float); pva=np.asarray(m.predict(Pool(Xv,cat_features=idx)),float)
    center=float(np.mean(ptr)) if len(ptr) else 0.0
    return ptr-center,pva-center,center
