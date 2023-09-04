from embeddings import making_vector

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime


def norm_border(dlen):
    # функция с горизонтальной асимптотой, которая выдает значения от 0 до 2.6 (для релевантных значений от 2 до inf)
    # с ее помощью нормализация будет происходить с переменными границами:
    # для портфелей с маленьким количеством сделок разброс значений будет маленький
    # для портфелей с больщим количеством в пределах 2.6
    return -np.exp(1)**(-0.03*dlen+1) + 2.6


def concat_deals(data, dataset, norm_parameter, use_flex):
    deals = pd.DataFrame([])
    for id in dataset.id:
        table = data[id]['deals']
        if table is not None:

            vec = making_vector(data[id]['deals'])
            if len(vec) > 1:
                # нормализуем вектор (множитель - гиперпараметр)
                if use_flex:
                    vec = ((vec - min(vec))/(max(vec)-min(vec))-0.5) * \
                        norm_border(len(table))
                else:
                    vec = ((vec - min(vec))/(max(vec)-min(vec))-0.5) * \
                        norm_parameter

            table['rclass'] = dataset[dataset.id == id]['class'].values + vec

            table["id"] = id  # добавляем id для соединения дилсов с юзером

            deals = pd.concat([deals, table], ignore_index=True) if len(
                deals) != 0 else table  # конкатим все дилсы в один датасет
        else:
            # если таблиц deals нет, добавляем дефолтную строку
            deals.loc[deals.shape[0]] = [datetime.strptime(
                "2022-09-22 14:40:00.000", '%Y-%m-%d %H:%M:%S.%f'), 0, 0, 0, id, *dataset[dataset.id == id]['class'].values]

    # мерджим датасет со сделками по id
    dataset = pd.merge(deals, dataset, on='id')
    return dataset


def droping_col(dataset):
    dataset['datetime'] = pd.to_datetime(dataset['datetime'])
    dataset['datetime'] = dataset['datetime'].apply(lambda x: x.timestamp())
    return dataset.drop(["class", "rclass"], 1)


def split_with_deals(data, train, flex, norm_parameter, train_size):
    # делим на train и test
    X = train.drop(columns=["nickname"])
    y = train["class"]
    X_train, X_test, y_train1, y_test = train_test_split(
        X, y, random_state=42, train_size=train_size)

    # добавляем сделки в датасет
    X_train = concat_deals(data, X_train, norm_parameter, flex)
    X_test = concat_deals(data, X_test, norm_parameter, flex)

    y_train = X_train.rclass

    # удаляем лишние признаки
    X_train = droping_col(X_train)
    X_test = droping_col(X_test)

    return X_train, X_test, y_train, y_test


def my_train_test_split(data, train, use_flex, train_size=0.75, with_deals=0, norm_par=1):
    if with_deals:
        X_train, X_test, y_train, y_test = split_with_deals(
            data, train, use_flex, norm_par, train_size)
    else:
        X = train.drop(columns=["class", "nickname", "id"])
        y = train["class"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=42, train_size=train_size)

    return X_train, X_test, y_train, y_test
