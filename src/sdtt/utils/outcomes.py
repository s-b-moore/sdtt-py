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

    # list of values to assign based on conditions
    values = ["hit", "miss", "correct_rejection", "false_alarm"]

    # creating new column and assigning values based on conditions
    dat["outcome"] = np.select(conditions, values, np.dtypes.Int64DType)

    return dat
