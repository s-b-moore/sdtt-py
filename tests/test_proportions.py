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
dat = pd.read_csv("data/example_out.csv")
dfc = dat.copy()

# --- calculate proportions ignoring the 'sequence' condition
# get counts for each outcome
df = pd.DataFrame({"count": dat.groupby(["id", "outcome"]).size()}).reset_index()
df = df.pivot(columns = "outcome", index = "id").reset_index()
df = df.fillna(0)

# proportions for each outcome
df["proportion", "hit"] = df["count", "hit"] / (df["count", "hit"] + df["count", "miss"])
df["proportion", "miss"] = df["count", "miss"] / (df["count", "miss"] + df["count", "hit"])
df["proportion", "correct_rejection"] = df["count", "correct_rejection"] / (df["count", "correct_rejection"] + df["count", "false_alarm"])
df["proportion", "false_alarm"] = df["count", "false_alarm"] / (df["count", "false_alarm"] + df["count", "correct_rejection"])

# --- calculate proportions including the 'sequence' condition
# get counts for each outcome
dfc = pd.DataFrame({"count": dat.groupby(["id", "sequence", "outcome"]).size()}).reset_index()
dfc = dfc.pivot(columns = "outcome", index = ["id", "sequence"]).reset_index()
dfc = dfc.fillna(0)

# proportions for each outcome
dfc["proportion", "hit"] = dfc["count", "hit"] / (dfc["count", "hit"] + dfc["count", "miss"])
dfc["proportion", "miss"] = dfc["count", "miss"] / (dfc["count", "miss"] + dfc["count", "hit"])
dfc["proportion", "correct_rejection"] = dfc["count", "correct_rejection"] / (dfc["count", "correct_rejection"] + dfc["count", "false_alarm"])
dfc["proportion", "false_alarm"] = dfc["count", "false_alarm"] / (dfc["count", "false_alarm"] + dfc["count", "correct_rejection"])

# set up class for testing
class TestProps(unittest.TestCase):

    # testing calculation of proportions ignoring condition
    def test_propsnc(self):
        nc_res = sdtt.props(dat, id = "id", outcome = "outcome", condition = None, correction = None)
        assert_frame_equal(nc_res, df)

    # testing calculation of proportions with condition
    def test_propscond(self):
        c_res = sdtt.props(dat, id = "id", outcome = "outcome", condition = "sequence", correction = None)
        assert_frame_equal(c_res, dfc)

if __name__ == "__main__":
    unittest.main()

