import pandas as pd
import numpy as np
from random import randint
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def random_embedding(vec, fix_size):
    emb = [vec[randint(0, len(vec)-1)] for _ in range(fix_size)]
    return emb


def sort_dist_embedding(vec, fix_size):
    '''
    making a fix_size embedding by selecting far values after vector sorting

    parameters: vec - pca vector, fix_size - size for embedding
    returns: emb - fix_size embedding
    '''

    emb = [sorted(vec)[i*(len(vec)//(fix_size-1)) if i != fix_size-1 else -1] for i in range(fix_size)]
    return emb


def distribution_discription_embedding(vec):
    emb = [np.mean(vec), np.std(vec), pd.Series(vec).skew(), pd.Series(vec).kurt()]
    return emb


def approximation_embedding(vec, fix_size):
    emb = [np.mean(sorted(vec)[(i-1)*(len(vec)//(fix_size-1)): (i*(len(vec)//(fix_size-1))
                                                                if i != fix_size-1 else -1)])
                                                                for i in range(1, fix_size+1)]

    emb = [i if not np.isnan(i) else 0 for i in emb]
    return emb


def making_vector(table):
    # prepro
    table = table.drop(columns=["ticker"])
    table['datetime'] = pd.to_datetime(table['datetime'])
    table['datetime'] = table['datetime'].apply(lambda x: x.timestamp())

    # PCA vector
    pca = PCA(n_components=1)
    scaler = StandardScaler().fit_transform(table)
    vector = pca.fit_transform(scaler)
    vector = [i[0] for i in vector]
    return vector


def adding_embeddings(dataset, data, fix_size=5):
    '''
    adding embedding of additional tables to main dataframe

    parameters: dataset - pd.DataFrame (dataset for adding features)
    returns: dataset - pd.DataFrame (dataset with new columns)
    '''

    for i in range(fix_size):
        dataset[f"emb{i}"] = 0

    for id in dataset.id:
        table = data[id]['deals']

        if table is not None:
            # vec
            vec = making_vector(table)

            # embs
            emb = sort_dist_embedding(vec, fix_size)

            # adding emb to dataset
            dataset.loc[dataset.id == id, "emb0":f"emb{fix_size-1}"] = emb

    return dataset
