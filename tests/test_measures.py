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
df_nc = pd.read_csv("data/example_proportions_nc.csv") # data for ignoring 'sequence' condition
df_c = pd.read_csv("data/example_proportions_c.csv") # data for including 'sequence' condition

# --- d'
# calculate d' ignoring 'sequence' condition
df_nc["d_prime"] = stats.norm.ppf(df_nc["p_hit"]) - stats.norm.ppf(df_nc["p_fa"])

# calculate d' including 'sequence' condition


# set up class for testing
class TestDprime(unittest.TestCase):

    # test calculation of d' ignoring condition
    def test_dnc(self):
        nc_res = sdtt.dprime(df_nc, hit_var = "p_hit", fa_var = "p_fa")
        assert_frame_equal(nc_res, df_nc)

if __name__ == "__main__":
    unittest.main()
