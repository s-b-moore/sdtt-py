# import required libraries
import pandas as pd
import numpy as np

# function to assign outcomes to binary data
def outcomes(data, accuracy_var, signal_var):
    dat = data.copy()

    # generate a list of conditions
    conditions = [
            (dat[accuracy_var] == 1) & (dat[signal_var] == 1),
            (dat[accuracy_var] == 0) & (dat[signal_var] == 1),
            (dat[accuracy_var] == 1) & (dat[signal_var] == 0),
            (dat[accuracy_var] == 0) & (dat[signal_var] == 0)
            ]

    # list of values
    values = ["hit", "miss", "correct_rejection", "false_alarm"]

    # create new column and assign values based on conditions
    dat["outcome"] = np.select(conditions, values, np.dtypes.Int64DType)

    return dat

# function to calculate proportions for each outcome
def proportions(data, id_var, outcome_var, condition_var = None, correction = None):
    dat = data.copy()

    # get counts for each outcome
    if condition_var == None:
        dat = pd.DataFrame(dat.groupby([id_var, outcome_var], as_index = False).size())
        dat = dat.pivot(columns = outcome_var, index = id_var, values = "size").reset_index()
        dat = dat.fillna(0) # outcomes with n=0 return na/nan
    else:
        dat = pd.DataFrame(dat.groupby([id_var, condition_var, outcome_var], as_index = False).size())
        dat = dat.pivot(columns = outcome_var, index = [id_var, condition_var], values = "size").reset_index()
        dat = dat.fillna(0)

    # calculate proportions
    dat["p_hit"] = dat["hit"] / (dat["hit"] + dat["miss"])
    dat["p_miss"] = dat["miss"] / (dat["miss"] + dat["hit"])
    dat["p_cr"] = dat["correct_rejection"] / (dat["correct_rejection"] + dat["false_alarm"])
    dat["p_fa"] = dat["false_alarm"] / (dat["false_alarm"] + dat["correct_rejection"])

    return dat
