import pandas as pd
import os

def additional_data_to_dict(mode: str, main_df: pd.DataFrame) -> dict:
    data = dict()
    ids = main_df.id
    for id in ids:
        acc_conf = f'data/{mode}/{mode}_additional_info/id_{id}/account_condition_{id}.csv'
        ref_point = f'data/{mode}/{mode}_additional_info/id_{id}/reference_point_{id}.csv'
        stat_table = f'data/{mode}/{mode}_additional_info/id_{id}/stats_table_{id}.csv'
        deals_1 = f'data/{mode}/{mode}_deals/1_{id}.csv'
        deals_2 = f'data/{mode}/{mode}_deals/2_{id}.csv'
        deals_3 = f'data/{mode}/{mode}_deals/3_{id}.csv'
        data[id] = {
                  'account_condition': pd.read_csv(acc_conf) if os.path.exists(acc_conf) else None, 
                  'reference_point': pd.read_csv(ref_point) if os.path.exists(ref_point) else None, 
                  'stats_table': pd.read_csv(stat_table, sep=';') if os.path.exists(stat_table) else None,
                  'deals': pd.read_csv(deals_1, index_col=0) if os.path.exists(deals_1) 
                                else pd.read_csv(deals_2, index_col=0) if os.path.exists(deals_2)
                                else pd.read_csv(deals_3, index_col=0) if os.path.exists(deals_3)
                                else None           
                }
    return data

def drop_outlier(df: pd.DataFrame, lower_quantile: float, upper_quantile: float):
    columns = ['start_sum', 'income_percent', 'deals']

    # Словари для хранения нижних и верхних границ для каждого столбца
    lower_bounds = {}
    upper_bounds = {}

    # Вычисление нижних и верхних границ для каждого столбца
    for col in columns:
        Q1 = df[col].quantile(lower_quantile)
        Q3 = df[col].quantile(upper_quantile)
        IQR = Q3 - Q1
        lower_bound = Q1 - 2.0 * IQR
        upper_bound = Q3 + 2.0 * IQR
        lower_bounds[col] = lower_bound
        upper_bounds[col] = upper_bound

    # Создаем фильтр для удаления выбросов по вычисленным границам
    mask = pd.Series(True, index=df.index)
    for col in columns:
        mask = mask & (df[col] >= lower_bounds[col]) & (df[col] <= upper_bounds[col])

    # Применяем фильтр к датафрейму
    df_filtered = df[mask]
    df = df_filtered
    return df


