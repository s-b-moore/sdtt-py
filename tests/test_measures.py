# import required libraries
import sys
import pandas as pd
from pandas.testing import assert_frame_equal
from scipy import stats
import unittest
from src.sdtt import measures

# read in data for testing
df_nc = pd.read_csv("data/example_props_nc.csv") # data for ignoring 'sequence' condition
df_c = pd.read_csv("data/example_props_c.csv") # data for including 'sequence' condition

# --- d'
# calculate d' ignoring 'sequence' condition
df_nc["d_prime"] = stats.norm.ppf(df_nc["p_hit"]) - stats.norm.ppf(df_nc["p_fa"])

# calculate d' including 'sequence' condition
df_c["d_prime"]

# set up class for testing
class TestDprime(unittest.TestCase):

    # test calculation of d' ignoring condition
    def test_dnc(self):
        nc_res = sdtt.dprime(df, hit_var = "p_hit", fa_var = "p_fa")
        assert_frame_equal(nc_res, df_nc)

if __name__ == "__main__":
    unittest.main()
