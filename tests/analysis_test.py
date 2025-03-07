"""
This script takes the underlying functionality of the sdtt package and combines it into a single
analysis script. The primary purpose of this is to generate .csv files using the functionality
within sdtt which is subsequently used in the unit tests you see in the current directory.
Thus, should any changes be made to the code underlying the functions within the sdtt package,
these will be updated here to ensure the data outputs are accurate, with tests also being updated
to reflect any changes made.
"""

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

# NOTE: comment out if not needed
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

# NOTE: comment out if not needed
# write the new data to csv
#prop_dat.to_csv("data/example_proportions_nc.csv", index = False)

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

# NOTE: comment out if not needed
# NOTE: this data contains extreme proportions and will return inf for some measures and prevent calculation of others
# write the new data to csv
#cprop_dat.to_csv("data/example_proportions_c.csv", index = False)

# --- check proportions to ensure no extreme values are present
# NOTE: extreme values are only present in the conditional example data provided (cprop_dat)
ext_props = cprop_dat.copy()

# simplest method is to check for inf values in d'
ext_props["d_prime"] = stats.norm.ppf(ext_props["p_hit"]) - stats.norm.ppf(ext_props["p_fa"])

if np.isinf(ext_props["d_prime"]).values.any():
    print("Extreme values detected. Recalculating proportions...")
    ext_dat = ext_props.copy()
    ext_dat["hit"] = ext_dat["hit"] + 0.5
    ext_dat["miss"] = ext_dat["miss"] + 0.5
    ext_dat["correct_rejection"] = ext_dat["correct_rejection"] + 0.5
    ext_dat["false_alarm"] = ext_dat["false_alarm"] + 0.5
    ext_dat["p_hit"] = ext_dat["hit"] / (ext_dat["hit"] + ext_dat["miss"])
    ext_dat["p_miss"] = ext_dat["miss"] / (ext_dat["miss"] + ext_dat["hit"])
    ext_dat["p_cr"] = ext_dat["correct_rejection"] / (ext_dat["correct_rejection"] + ext_dat["false_alarm"])
    ext_dat["p_fa"] = ext_dat["false_alarm"] / (ext_dat["false_alarm"] + ext_dat["correct_rejection"])
else:
    print("No extreme values detected.")

# recalculate d' to test
ext_dat["d_prime"] = stats.norm.ppf(ext_dat["p_hit"]) - stats.norm.ppf(ext_dat["p_fa"])

ext_dat = ext_dat.drop(columns = ["d_prime"])

# NOTE: comment out if not needed
# write the new data to csv
ext_dat.to_csv("data/example_corrected.csv", index = False)

# --- calculate measures (using corrected data)
# d'
d_dat = ext_dat.copy()
d_dat["d_prime"] = stats.norm.ppf(d_dat["p_hit"]) - stats.norm.ppf(d_dat["p_fa"])

# NOTE: comment out if not needed
# write the new data to csv
#d_dat.to_csv("data/example_dprime.csv", index = False)

# criterion location
c_dat = ext_dat.copy()
c_dat["criterion"] = -0.5 * (stats.norm.ppf(c_dat["p_hit"]) + stats.norm.ppf(c_dat["p_fa"]))

# NOTE: comment out if not needed
# write the new data to csv
#c_dat.to_csv("data/example_criterion.csv", index = False)

# c' (relative criterion location)
cpr_dat = ext_dat.copy()
cpr_dat["c_prime"] = -0.5 * (stats.norm.ppf(cpr_dat["p_hit"]) + stats.norm.ppf(cpr_dat["p_fa"])) / (stats.norm.ppf(cpr_dat["p_hit"]) - stats.norm.ppf(cpr_dat["p_fa"]))

# NOTE: comment out if not needed
# write the new data to csv
#cpr_dat.to_csv("data/example_cprime.csv", index = False)

# likelihood ratio (beta)
b_dat = ext_dat.copy()
b_dat["lr_beta"] = -0.5 * (stats.norm.ppf(b_dat["p_hit"])**2 - stats.norm.ppf(b_dat["p_fa"])**2)

# NOTE: comment out if not needed
# write the new data to csv
#b_dat.to_csv("data/example_lrbeta.csv", index = False)
