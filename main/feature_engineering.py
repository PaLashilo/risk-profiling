from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np


def adding_deal_sum_info(df: pd.DataFrame, data: dict) -> pd.DataFrame:
    df['mean_deal_sum'] = 0
    df['std_deal_sum'] = 0
    df['len_deals'] = 0
    for id in df.id:
        table = data[id]['deals']
        if table is not None:
            df.mean_deal_sum.loc[df.id == id] = table.summ.mean()
            std = table.summ.std()
            df.std_deal_sum.loc[df.id == id] = std if not np.isnan(std) else 0
            df.len_deals.loc[df.id == id] = len(table)

    non_zero_rows = df[df['mean_deal_sum'] != 0].drop(columns=["nickname", "id", "std_deal_sum", "class"])
    lin_reg = LinearRegression()
    lin_reg.fit(non_zero_rows.drop(columns=['mean_deal_sum']), non_zero_rows['mean_deal_sum'])
    zero_rows = df[df['mean_deal_sum'] == 0].drop(columns=["nickname", "id", "std_deal_sum", "class"])
    predicted_values = lin_reg.predict(zero_rows.drop(columns=['mean_deal_sum']))
    df.loc[df['mean_deal_sum'] == 0, 'mean_deal_sum'] = predicted_values

    non_zero_rows = df[df['std_deal_sum'] != 0].drop(columns=["nickname", "id", "class"])
    lin_reg = LinearRegression()
    lin_reg.fit(non_zero_rows.drop(columns=['std_deal_sum']), non_zero_rows['std_deal_sum'])
    zero_rows = df[df['std_deal_sum'] == 0].drop(columns=["nickname", "id", "class"])
    predicted_values = lin_reg.predict(zero_rows.drop(columns=['std_deal_sum']))
    df.loc[df['std_deal_sum'] == 0, 'std_deal_sum'] = predicted_values

    return df


def adding_frequent_market_info(df: pd.DataFrame, data: dict) -> pd.DataFrame:
    df['stock_coef'] = 0
    df['curr_coef'] = 0
    df['fort_coef'] = 0
    # df['freq_market'] = 0

    df['income_percent_stats'] = 0

    for id in df.id:
        table = data[id]['stats_table']
        if table is not None:

            # market coefs
            stock_turnover = table.loc[1, 'part_turnover'] if table.loc[1, 'part_turnover'] != '-' else 0
            curr_turnover = table.loc[3, 'part_turnover'] if table.loc[3, 'part_turnover'] != '-' else 0
            fort_turnover = table.loc[2, 'part_turnover'] if table.loc[2, 'part_turnover'] != '-' else 0

            df.stock_coef.loc[df.id == id] = float(stock_turnover)
            df.curr_coef.loc[df.id == id] = float(curr_turnover)
            df.fort_coef.loc[df.id == id] = float(fort_turnover)

            # df.freq_market.loc[df.id == id] = \
            #             max(enumerate([0, float(stock_turnover), float(curr_turnover), float(fort_turnover)]),key=lambda x: x[1])[0]

            # income
            df.income_percent_stats.loc[df.id == id] = table.loc[0, "income_percent"]

    return df.fillna(0)


def adding_free_funds_info(df: pd.DataFrame, data: dict) -> pd.DataFrame:
    df['std_free_funds'] = 0
    df['coef_free_funds'] = 0
    df['coef_free_funds2'] = 0
    df['mean_free_funds'] = 0
    df['min_free_funds'] = 0
    df['max_free_funds'] = 0

    for id in df.id:
        table = data[id]['account_condition']
        if table is not None:
            std = (table.free_funds).std()
            df.std_free_funds.loc[df.id == id] = std

            if len(table) < 1:
                df.coef_free_funds.loc[df.id == id] = 0
                df.coef_free_funds2.loc[df.id == id] = 0
            else:
                df.coef_free_funds.loc[df.id == id] = std / table.start_sum.iloc[0]
                df.coef_free_funds2.loc[df.id == id] = (std / table.start_sum.iloc[0])**2

            df.mean_free_funds.loc[df.id == id] = np.mean(table.free_funds)
            df.max_free_funds.loc[df.id == id] = np.max(table.free_funds)
            df.min_free_funds.loc[df.id == id] = np.min(table.free_funds)

    return df


def adding_features(df: pd.DataFrame, data: dict) -> pd.DataFrame:
    df = adding_deal_sum_info(df, data)
    df = adding_frequent_market_info(df, data)
    df = adding_free_funds_info(df, data)
    return df
