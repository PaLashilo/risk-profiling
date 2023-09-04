import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score


def calculate_metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    f1 = f1_score(y_true, y_pred, average='micro')
    return rmse, f1


def model_predict(model, X_train, X_test):
    y_pred_ts = model.predict(X_test)
    y_pred_tr = model.predict(X_train)
    return y_pred_ts, y_pred_tr


def y_downsampling(model_predictions, data, dataset):
    id_len = {id: len(data[id]['deals'] if data[id]['deals']
                      is not None else [0]) for id in dataset.id}
    y_pred = []
    i = 0
    for id in dataset.id.unique():

        # считаем среднее по классам юзера
        mean_y = np.mean(model_predictions[i:i+id_len[id]])

        # обработка выхода за границу
        if mean_y > 5:
            mean_y = 5
        if mean_y < 1:
            mean_y = 1

        # добавляем итоговый класс юзера (средний от всех)
        y_pred = np.append(y_pred, round(mean_y))
        i += id_len[id]

    return y_pred


def metrics(cbc, X_train, X_test, y_train, y_test, data, with_deals=0):
    '''
    calculation of metrics f1-score and RMSE

    parameters: y_true - series of right classes, y_pred - series of predicted classes
    returns: rmse, f1 - tuple of metrics
    '''
    y_pred_ts, y_pred_tr = model_predict(cbc, X_train, X_test)

    if with_deals:
        metrics_tr = calculate_metrics(y_downsampling(
            y_train, X_train, data), y_downsampling(y_pred_tr, X_train, data))
        metrics_ts = calculate_metrics(
            y_test, y_downsampling(y_pred_ts, X_test, data))

    else:
        metrics_tr = calculate_metrics(y_train, y_pred_tr)
        metrics_ts = calculate_metrics(y_test, y_pred_ts)

    print(
        f"TRAIN: \nrmse:{metrics_tr[0]}\nf1: {metrics_tr[1]}\nTEST: \nrmse:{metrics_ts[0]}\nf1: {metrics_ts[1]}")
    return metrics_tr, metrics_ts
