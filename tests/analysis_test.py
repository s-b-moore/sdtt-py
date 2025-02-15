# import required libraries
import pandas as pd
import numpy as np
from scipy import stats

# read in data for testing
dat = pd.read_csv("data/example_data.csv")
out_dat = dat.copy()

# --- assign outcomes
# generate a list of conditions
conditions = [
        (out_dat["accuracy"] == 1) & (out_dat["change_type"] == 1),
        (out_dat["accuracy"] == 0) & (out_dat["change_type"] == 1),
        (out_dat["accuracy"] == 1) & (out_dat["change_type"] == 0),
        (out_dat["accuracy"] == 0) & (out_dat["change_type"] == 0)
        ]

# list of values to assign based on conditions
values = ["hit", "miss", "correct_rejection", "false_alarm"]

# creating new column and assigning values based on outcome
out_dat["outcome"] = np.select(conditions, values, np.dtypes.Int64DType)

# write the new data to csv
#out_dat.to_csv("data/example_outcome.csv", index = False)

# --- calculate proportions ignoring the 'sequence' condition
# get counts for each outcome
prop_dat = out_dat.copy()
prop_dat = pd.DataFrame(prop_dat.groupby(["id", "outcome"], as_index = False).size())
prop_dat = prop_dat.pivot(columns = "outcome", index = "id", values = "size").reset_index()
prop_dat = prop_dat.fillna(0)

# proportions for each outcome
prop_dat["p_hit"] = prop_dat["hit"] / (prop_dat["hit"] + prop_dat["miss"])
prop_dat["p_miss"] = prop_dat["miss"] / (prop_dat["miss"] + prop_dat["hit"])
prop_dat["p_cr"] = prop_dat["correct_rejection"] / (prop_dat["correct_rejection"] + prop_dat["false_alarm"])
prop_dat["p_fa"] = prop_dat["false_alarm"] / (prop_dat["false_alarm"] + prop_dat["correct_rejection"])

# write the new data to csv
prop_dat.to_csv("data/example_proportions_nc.csv", index = False)

# --- calculate proportions including the 'sequence' condition
cprop_dat = out_dat.copy()
cprop_dat = pd.DataFrame(cprop_dat.groupby(["id", "sequence", "outcome"], as_index = False).size())
cprop_dat = cprop_dat.pivot(columns = "outcome", index = ["id", "sequence"], values = "size").reset_index()
cprop_dat = cprop_dat.fillna(0)

# proportions for each outcome
cprop_dat["p_hit"] = cprop_dat["hit"] / (cprop_dat["hit"] + cprop_dat["miss"])
cprop_dat["p_miss"] = cprop_dat["miss"] / (cprop_dat["miss"] + cprop_dat["hit"])
cprop_dat["p_cr"] = cprop_dat["correct_rejection"] / (cprop_dat["correct_rejection"] + cprop_dat["false_alarm"])
cprop_dat["p_fa"] = cprop_dat["false_alarm"] / (cprop_dat["false_alarm"] + cprop_dat["correct_rejection"])

# write the new data to csv
cprop_dat.to_csv("data/example_proportions_c.csv", index = False)

# --- calculate measures
# d'
d_dat = cprop_dat.copy()
d_dat["d_prime"] = stats.norm.ppf(d_dat["p_hit"]) - stats.norm.ppf(d_dat["p_fa"])

# write the new data to csv
d_dat.to_csv("data/example_dprime.csv", index = False)

# criterion location
c_dat = cprop_dat.copy()
c_dat["criterion"] = -0.5 * (stats.norm.ppf(c_dat["p_hit"]) + stats.norm.ppf(c_dat["p_fa"]))

# write the new data to csv
c_dat.to_csv("data/example_criterion.csv", index = False)

# c' (relative criterion location)
cpr_dat = cprop_dat.copy()
cpr_dat["c_prime"] = -0.5 * (stats.norm.ppf(cpr_dat["p_hit"]) + stats.norm.ppf(cpr_dat["p_fa"])) / (stats.norm.ppf(cpr_dat["p_hit"]) - stats.norm.ppf(cpr_dat["p_fa"]))

# write the new data to csv
cpr_dat.to_csv("data/example_cprime.csv", index = False)

# likelihood ratio (beta)
b_dat = cprop_dat.copy()
b_dat["lr_beta"] = -0.5 * (stats.norm.ppf(b_dat["p_hit"])**2 - stats.norm.ppf(b_dat["p_fa"])**2)

# write the new data to csv
b_dat.to_csv("data/example_lrbeta.csv", index = False)

# TODO: implement checking function for extreme values
