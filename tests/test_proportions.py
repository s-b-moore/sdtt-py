# import required libraries
import sys
import pandas as pd
from pandas.testing import assert_frame_equal
import unittest

# set the path to import the sdtt module
par_dir = ".."
sys.path.append(par_dir)

# import sdtt module
import sdtt

# read in data for testing
dat = pd.read_csv("data/example_outcome.csv")
dfc = dat.copy()

# --- calculate proportions ignoring the 'sequence' condition
# get counts for each outcome
df = pd.DataFrame(dat.groupby(["id", "outcome"], as_index = False).size())
df = df.pivot(columns = "outcome", index = "id", values = "size").reset_index()
df = df.fillna(0)

# proportions for each outcome
df["p_hit"] = df["hit"] / (df["hit"] + df["miss"])
df["p_miss"] = df["miss"] / (df["miss"] + df["hit"])
df["p_cr"] = df["correct_rejection"] / (df["correct_rejection"] + df["false_alarm"])
df["p_fa"] = df["false_alarm"] / (df["false_alarm"] + df["correct_rejection"])

# --- calculate proportions including the 'sequence' condition
# get counts for each outcome
dfc = pd.DataFrame(dat.groupby(["id", "sequence", "outcome"], as_index = False).size())
dfc = dfc.pivot(columns = "outcome", index = ["id", "sequence"], values = "size").reset_index()
dfc = dfc.fillna(0)

# proportions for each outcome
dfc["p_hit"] = dfc["hit"] / (dfc["hit"] + dfc["miss"])
dfc["p_miss"] = dfc["miss"] / (dfc["miss"] + dfc["hit"])
dfc["p_cr"] = dfc["correct_rejection"] / (dfc["correct_rejection"] + dfc["false_alarm"])
dfc["p_fa"] = dfc["false_alarm"] / (dfc["false_alarm"] + dfc["correct_rejection"])

# set up class for testing
class TestProps(unittest.TestCase):

    # testing calculation of proportions ignoring condition
    def test_propsnc(self):
        nc_res = sdtt.proportions(dat, id_var = "id", outcome_var = "outcome", condition_var = None, correction = None)
        assert_frame_equal(nc_res, df)

    # testing calculation of proportions with condition
    def test_propscond(self):
        c_res = sdtt.proportions(dat, id_var = "id", outcome_var = "outcome", condition_var = "sequence", correction = None)
        assert_frame_equal(c_res, dfc)

if __name__ == "__main__":
    unittest.main()

