# required libraries
import pandas as pd
import numpy as np
from scipy import stats

# function to assign outcomes
def outcomes(data, accuracy, signal):    
    dat = data

    # make a list of conditions here to use with values later
    conditions = [
            (dat[accuracy] == 1) & (dat[signal] == 1) # hit
            (dat[accuracy] == 0) & (dat[signal] == 1) # miss
            (dat[accuracy] == 1) & (dat[signal] == 0) # correct rejection
            (dat[accuracy] == 0) & (dat[signal] == 0) # false alarm
            ]

    # values to assign based on conditions above
    values = ["hit", "miss", "correct_rejection", "false_alarm"]

    # create new column and use 'np.select' to assign values
    dat["outcome"] = np.select(conditions, values, np.dtypes.Int64Dtype)

    # return data frame
    return dat

# function to get proportions of outcomes
def props(data, id, outcome, condition = None, correction = None):
    dat = data

    # if/else to calculate counts based on condition paramater
    if condition is None:
        df = pd.DataFrame({"count": dat.groupby([id, outcome]).size()}).reset_index()
        df = df.pivot(columns = outcome, index = id).reset_index()
        df = df.fillna(0)
    else:
        df = pd.DataFrame({"count": dat.groupby([id, condition, outcome]).size}).reset_index()
        df = df.pivot(columns = outcome, index = [id, condition]).reset_index()
        df = df.fillna(0)

    # proportions are calculated here
    df["proportions", "p_cr"] = df["count", "correct_rejection"] / (df["count", "correct_rejection"] + df["count", "false_alarm"])
    df["proportions", "p_fa"] = df["count", "false_alarm"] / (df["count", "false_alarm"] + df["count", "correct_rejection"])
    df["proportions", "p_hit"] = df["count", "hit"] / (df["count", "hit"] + df["count", "miss"])
    df["proportions", "p_miss"] = df["count", "miss"] / (df["count", "miss"] + df["count", "hit"])

    # return data
    return df
