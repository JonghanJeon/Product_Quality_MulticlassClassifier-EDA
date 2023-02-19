import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def preprocess(train_X, test):
    """ 샘플이 적거나 결측치가 많은 열을 제거

    Parameters
    ----------
    train_X : df.DataFrame
        학습할 데이터셋의 데이터프레임

    test : df.DataFrame
        추론할 데이터셋의 데이터프레임

    Returns
    -------
    train_X : df.DataFrame
        인코딩된 학습 데이터프레임
    test : df.DataFrame
        인코딩된 추론 데이터프레임
    
    Notes
    -----
    의미없는 컬럼을 제거
    """
    
    x_feat = np.array([x for x in train_X.columns if 'X_' in x])
    cos_sim = cosine_similarity(train_X[x_feat].fillna(0).transpose())
    threshold = np.where(cos_sim>0.99, True, False)

    graph = {}
    for i, row in enumerate(threshold):
        row = np.where(row==True)[0]
        is_in = False
        for k,v in graph.items():
            if len(set(row)&set(v))!=0:
                graph[k]=set(row)|set(v)
                is_in=True
        if is_in==False:
            graph[i]=set(row)
    
    res = {}
    del_keys = []

    for t in graph.keys():
        if t not in res.keys():
            res[t]=graph[t]
        for k,v in list(graph.items()):
            if t>=k:
                continue
            if len(set(graph[t])&set(v))!=0:
                res[t]=set(res[t])|set(v)
                del_keys.append(k)
    for k in del_keys:
        res.pop(k, None)

    for k,v in res.items():
        groups = x_feat[list(v)]
        new_col = "X_%d_mean"%(k+1)
        train_X[new_col] = train_X[groups].mean(axis=1)
        #train_X=train_X.drop(groups, axis=1)
        test[new_col] = test[groups].mean(axis=1)
        #test=test.drop(groups, axis=1)

    return train_X, test