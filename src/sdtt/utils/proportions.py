# import required libraries
import pandas as pd

# function to calculate proportions
def proportions(data, id_var, outcome_var, condition_var = None, correction = None):

    dat = data.copy()

    # get counts for each outcome
    if condition_var == None:
        p_dat = pd.DataFrame(dat.groupby([id_var, outcome_var], as_index = False).size())
        p_dat = p_dat.pivot(columns = outcome_var, index = id_var, values = "size").reset_index()
        p_dat = p_dat.fillna(0) # outcomes with n = 0 return na/nan
    else:
        p_dat = pd.DataFrame(dat.groupby([id_var, condition_var, outcome_var], as_index = False).size())
        p_dat = p_dat.pivot(columns = outcome_var, index = [id_var, condition_var], values = "size").reset_index()
        p_dat = p_dat.fillna(0)

    # calculate proportions
    p_dat["p_hit"] = p_dat["hit"] / (p_dat["hit"] + p_dat["miss"])
    p_dat["p_miss"] = p_dat["miss"] / (p_dat["miss"] + p_dat["hit"])
    p_dat["p_cr"] = p_dat["correct_rejection"] / (p_dat["correct_rejection"] + p_dat["false_alarm"])
    p_dat["p_fa"] = p_dat["false_alarm"] / (p_dat["false_alarm"] + p_dat["correct_rejection"])

    return p_dat
