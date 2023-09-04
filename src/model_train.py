from catboost import CatBoostClassifier
from catboost import CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV


def model_training(X_train, y_train, param_grid, random_iter, with_deals=0):
    # cat_columns = X_train.select_dtypes(include=['object']).columns

    if with_deals:
        model = CatBoostRegressor(verbose=0, loss_function='RMSE')
        score = 'neg_mean_squared_error'
    else:
        model = CatBoostClassifier(verbose=0, loss_function='MultiClass')
        score = 'f1_macro'

    grid_search = RandomizedSearchCV(estimator=model,
                                     param_distributions=param_grid,
                                     cv=5,
                                     scoring=score,
                                     n_iter=random_iter)

    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    print("Best CV Score:", grid_search.best_score_)

    results = grid_search.cv_results_
    print("Results of all configurations", results['mean_test_score'])

    return grid_search
