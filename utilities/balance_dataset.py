from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def over_sampling_cat(x, y, seed, strategy):
    ros = RandomOverSampler(random_state=seed, sampling_strategy=strategy)
    x_res, y_res = ros.fit_resample(x.reshape(-1, 1), y)
    return x_res, y_res

def under_sampling_cat(x, y, seed, strategy):
    rus = RandomUnderSampler(random_state=seed, sampling_strategy=strategy)
    x_res, y_res = rus.fit_resample(x.reshape(-1, 1), y)
    return x_res, y_res

def combined_cat_sampling(x, y, seed, over_strategy, under_strategy):
    x_res, y_res = over_sampling_cat(x, y, seed, over_strategy)
    x_res, y_res = under_sampling_cat(x_res, y_res, seed, under_strategy)
    return x_res, y_res

