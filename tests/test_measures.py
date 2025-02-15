# import required libraries
import sys
import pandas as pd
from pandas.testing import assert_frame_equal
from scipy import stats
import unittest

# set the path to import the sdtt module
par_dir = ".."
sys.path.append(par_dir)

# import sdtt module
import sdtt

# read in data for testing
df = pd.read_csv("data/example_props_nc.csv") # data for ignoring 'sequence' condition
dfc = pd.read_csv("data/example_props_c.csv") # data for including 'sequence' condition

# --- d'
d_df = df.copy()

d_df["measure", "d_prime"] = stats.norm.ppf(d_df["proportion", "hit"]) - stats.norm.ppf(d_df["proportion", "false_alarm"])

# set up class for testing
class TestDprime(unittest.TestCase):

    # test calculation of d' ignoring condition
    def test_dnc(self):
        nc_res = sdtt.dprime(df, ["proportion", "hit"], ["proportion", "miss"])
        assert_frame_equal(nc_res, d_df)



if __name__ == "__main__":
    unittest.main()
