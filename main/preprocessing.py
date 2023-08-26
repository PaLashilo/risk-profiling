import os
import pandas as pd
import numpy as np


def format_account_condition(id: int, mode: str) -> str or None:
    path = f'./Data/{mode}/{mode}_additional_info/id_{id}/account_condition_{id}.csv'
    try:
        account_condition = pd.read_csv(path, sep=';', index_col=0)
    except FileNotFoundError:
        return 'No Account Condition file for ID: {id}'
    for clmn in ['start_sum', 'warranty_provision', 'free_funds']:
        delt_name = f'{clmn}_delt'
        account_condition[clmn].loc[account_condition[clmn] == '-'] = np.nan
        if account_condition[clmn].isna().sum() != account_condition.shape[0]:
            account_condition[[clmn, delt_name]] = account_condition[clmn].str.split('(', expand=True)
            account_condition[delt_name] = account_condition[delt_name].str.replace(')', '')
            account_condition[delt_name] = account_condition[delt_name].str.replace('+', '')
            account_condition[delt_name].loc[account_condition[delt_name] == '-'] = np.nan
        else:
            account_condition[delt_name] = np.nan
    astypes = {'start_sum': 'float',
               'warranty_provision': 'float',
               'free_funds': 'float',
               'start_sum_delt': 'float',
               'warranty_provision_delt': 'float',
               'free_funds_delt': 'float'}
    account_condition = account_condition.astype(astypes)
    os.remove(path)
    account_condition.to_csv(path)
    return None


def format_reference_point(id: int, mode: str) -> str or None:
    path = f'./Data/{mode}/{mode}_additional_info/id_{id}/reference_point_{id}.csv'
    try:
        reference_point = pd.read_csv(path, sep=';', index_col=0)
    except FileNotFoundError:
        return 'No Reference Point file for ID: {id}'
    for clmn in ['market', 'ticker', 'open_positions', 'price', 'end_day_balance', 'estimated_cost']:
        reference_point[clmn].loc[reference_point[clmn] == '-'] = np.nan
    new_clmn = 'open_positions_delt'
    if reference_point['open_positions'].isna().sum() != reference_point.shape[0]:
        reference_point[['open_positions', new_clmn]] = reference_point['open_positions'].str.split('(', expand=True)
        reference_point[new_clmn] = reference_point[new_clmn].str.replace(')', '')
        reference_point[new_clmn] = reference_point[new_clmn].str.replace('+', '')
        reference_point[new_clmn].loc[reference_point[new_clmn] == '-'] = np.nan
    else:
        reference_point[new_clmn] = np.nan
    astypes = {'open_positions': 'float',
               'price': 'float',
               'end_day_balance': 'float',
               'estimated_cost': 'float'}
    reference_point = reference_point.astype(astypes)
    os.remove(path)
    reference_point.to_csv(path)
    return reference_point


def format_deals(id: int, market: int, mode: str) -> str or None:
    path = f'./Data/{mode}/{mode}_deals/{market}_{id}.csv'
    try:
        deals = pd.read_csv(path, names=["datetime", "ticker", "quantity", "summ"], sep=';')
    except FileNotFoundError:
        return 'No deals data file for ID: {market}_{id}'
    for ticker in deals['ticker'].unique():
        search_value = ticker
        first_occurrence = deals[deals['ticker'] == search_value].iloc[0]
        last_occurrence = deals[deals['ticker'] == search_value].iloc[-1]
        start_price = abs(first_occurrence['summ']/first_occurrence['quantity'])
        final_price = abs(last_occurrence['summ']/last_occurrence['quantity'])
        volatility = ((final_price - start_price))/(start_price+0.001) * 100
        deals.replace(ticker, volatility, inplace=True)
    os.remove(path)
    deals.to_csv(path)
    return None

def format_main(mode: str, index_col: bool) -> None:
    path = f'./Data/{mode}/{mode}.csv'
    main = pd.read_csv(path, sep=',')
    for clmn in ['start_sum', 'income_rub', 'income_percent']:
        main[clmn] = main[clmn].astype(str)
        main[clmn] = main[clmn].str.replace(',', '.').str.replace(' ', '')
        # main[clmn] = pd.Series(main[clmn]).astype('str').replace(',', '.').astype('int')
        # main[clmn] = pd.Series(main[clmn]).astype('str').replace(' ', '').astype('int')
    main['income_percent'].loc[main['income_percent'] == '-'] = np.nan
    main = main.astype({'start_sum': 'float', 'income_rub': 'float', 'income_percent': 'float'})
    main['income_percent'] = main['income_percent'].fillna(main.income_rub.astype('int')/main.start_sum.astype('int'))
    os.remove(path)
    main.to_csv(path, index=index_col)
