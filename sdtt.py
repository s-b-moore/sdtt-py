# required libraries
import pandas as pd
import numpy as np
from scipy import stats

# function to assign outcomes
def outcomes(data, accuracy, signal):    
    dat = data.copy()

    # make a list of conditions here to use with values later
    conditions = [
            (dat[accuracy] == 1) & (dat[signal] == 1), # hit
            (dat[accuracy] == 0) & (dat[signal] == 1), # miss
            (dat[accuracy] == 1) & (dat[signal] == 0), # correct rejection
            (dat[accuracy] == 0) & (dat[signal] == 0) # false alarm
            ]

    # values to assign based on conditions above
    values = ["hit", "miss", "correct_rejection", "false_alarm"]

    # create new column and use 'np.select' to assign values
    dat["outcome"] = np.select(conditions, values, np.dtypes.Int64DType)

    # return data frame
    return dat

# function to get proportions of outcomes
def props(data, id, outcome, condition = None, correction = None):
    dat = data.copy()

    # if/else to calculate counts based on condition paramater
    if condition is None:
        df = pd.DataFrame({"count": dat.groupby([id, outcome]).size()}).reset_index()
        df = df.pivot(columns = outcome, index = id).reset_index()
        df = df.fillna(0)
    else:
        df = pd.DataFrame({"count": dat.groupby([id, condition, outcome]).size()}).reset_index()
        df = df.pivot(columns = outcome, index = [id, condition]).reset_index()
        df = df.fillna(0)

    # proportions are calculated here
    df["proportions", "hit"] = df["count", "hit"] / (df["count", "hit"] + df["count", "miss"])
    df["proportions", "miss"] = df["count", "miss"] / (df["count", "miss"] + df["count", "hit"])
    df["proportions", "correct_rejection"] = df["count", "correct_rejection"] / (df["count", "correct_rejection"] + df["count", "false_alarm"])
    df["proportions", "false_alarm"] = df["count", "false_alarm"] / (df["count", "false_alarm"] + df["count", "correct_rejection"])

    # return data
    return df

# TODO: implement overall function for calculations

# function to calculate d'
def dprime(data, hit_var, fa_var):
    dat = data.copy()

    dat["measure", "d_prime"] = stats.norm.ppf(dat[hit_var]) - stats.norm.ppf(dat[fa_var])

    return dat

# function to calculate criterion location
def criterion(data, hit_var, fa_var):
    dat = data.copy()

    dat["measure", "criterion_location"] = -0.5 * (stats.norm.ppf(dat[hit_var]) + stats.norm.ppf(dat[fa_var]))

    return dat





