# import required libraries
import pandas as pd
import numpy as np
from scipy import stats

# read in data for testing
dat = pd.read_csv("data/example_data.csv")

# --- assign outcomes
# generate a list of conditions
conditions = [
        (dat["accuracy"] == 1) & (dat["change_type"] == 1), # hit
        (dat["accuracy"] == 0) & (dat["change_type"] == 1), # miss
        (dat["accuracy"] == 1) & (dat["change_type"] == 0), # correct rejection
        (dat["accuracy"] == 0) & (dat["change_type"] == 0)  # false alarm
        ]

# generate a list of values to assign based on conditions above
values = ["hit", "miss", "correct_rejection", "false_alarm"]

# create new column and use 'np.select' to assign values
dat["outcome"] = np.select(conditions, values, np.dtypes.Int64DType)

# write the new data to csv
#dat.to_csv("data/example_out.csv", index = False)

# see the 'example_out.csv' file for the output of the above

# --- calculate proportions ignoring the 'sequence' condition
# get counts for each outcome
df = pd.DataFrame({"count": dat.groupby(["id", "outcome"]).size()}).reset_index()
df = df.pivot(columns = "outcome", index = "id").reset_index()
df = df.fillna(0)

# proportions for each outcome
df["proportions", "hit"] = df["count", "hit"] / (df["count", "hit"] + df["count", "miss"])
df["proportions", "miss"] = df["count", "miss"] / (df["count", "miss"] + df["count", "hit"])
df["proportions", "correct_rejection"] = df["count", "correct_rejection"] / (df["count", "correct_rejection"] + df["count", "false_alarm"])
df["proportions", "false_alarm"] = df["count", "false_alarm"] / (df["count", "false_alarm"] + df["count", "correct_rejection"])

# write the new data to csv
#df.to_csv("data/example_props_nc.csv", index = False)

# see the 'example_props_nc.csv' file for output of the above

# --- calculate proportions including the 'sequence' condition
# get counts for each outcome
df = pd.DataFrame({"count": dat.groupby(["id", "sequence", "outcome"]).size()}).reset_index()
df = df.pivot(columns = "outcome", index = ["id", "sequence"]).reset_index()
df = df.fillna(0)

# proportions for each outcome
df["proportions", "hit"] = df["count", "hit"] / (df["count", "hit"] + df["count", "miss"])
df["proportions", "miss"] = df["count", "miss"] / (df["count", "miss"] + df["count", "hit"])
df["proportions", "correct_rejection"] = df["count", "correct_rejection"] / (df["count", "correct_rejection"] + df["count", "false_alarm"])
df["proportions", "false_alarm"] = df["count", "false_alarm"] / (df["count", "false_alarm"] + df["count", "correct_rejection"])

# write the new data to csv
#df.to_csv("data/example_props_c.csv", index = False)

# --- calculate measures
# d'
dprime = df.copy()
dprime["measure", "d_prime"] = stats.norm.ppf(dprime["proportions", "hit"]) - stats.norm.ppf(dprime["proportions", "false_alarm"])

# write the new data to csv
#dprime.to_csv("data/example_dprime.csv", index = False)

# see the 'example_dprime.csv' file for output of the above

# criterion location
relcrit = df.copy()
relcrit["measure", "criterion"] = -0.5 * (stats.norm.ppf(relcrit["proportions", "hit"]) + stats.norm.ppf(relcrit["proportions", "false_alarm"]))

# write the new data to csv
relcrit.to_csv("data/example_relcrit.csv", index = False)

# see the 'example_relcrit.csv' file for output of the above
