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
dp_nc = df_nc.copy()
dp_nc["d_prime"] = stats.norm.ppf(dp_nc["p_hit"]) - stats.norm.ppf(dp_nc["p_fa"])

# calculate d' including 'sequence' condition
dp_c = df_c.copy()
dp_c["d_prime"] = stats.norm.ppf(dp_c["p_hit"]) - stats.norm.ppf(dp_c["p_fa"])

# set up class for testing d'
class TestDprime(unittest.TestCase):

    # test calculation of d' ignoring condition
    def test_dnc(self):
        dpt_nc = df_nc.copy()
        dpnc_res = sdtt.dprime(dpt_nc, hit_var = "p_hit", fa_var = "p_fa")
        assert_frame_equal(dpnc_res, dp_nc)

    # test calculation of d' including condition
    def test_dc(self):
        dpt_c = df_c.copy()
        dpc_res = sdtt.dprime(dpt_c, hit_var = "p_hit", fa_var = "p_fa")
        assert_frame_equal(dpc_res, dp_c)

# --- criterion (c)
# calculate criterion ignoring 'sequence' condition
cri_nc = df_nc.copy()
cri_nc["criterion"] = stats.norm.ppf(cri_nc["p_hit"] + stats.norm.ppf(cri_nc["p_fa"]))

# calculate criterion including 'sequence' condition
cri_c = df_c.copy()
cri_c["criterion"] = stats.norm.ppf(cri_c["p_hit"] + stats.norm.ppf(cri_c["p_fa"]))

# set up class for testing criterion
class TestCriterion(unittest.TestCase):

    # test calculation of criterion ignoring condition
    def test_cnc(self):
        crit_nc = df_nc.copy()
        cnc_res = sdtt.criterion(crit_nc, hit_var = "p_hit", fa_var = "p_fa")
        assert_frame_equal(cnc_res, cri_nc)

    def test_cc(self):
        crit_c = df_c.copy()
        cc_res = sdtt.criterion(crit_c, hit_var = "p_hit", fa_var = "p_fa")
        assert_frame_equal(cc_res, cri_c)

# --- c'
# calculate c' ignoring 'sequence' condition
cp_nc = df_nc.copy()
cp_nc["c_prime"] = -0.5 * (stats.norm.ppf(cp_nc["p_hit"]) + stats.norm.ppf(cp_nc["p_fa"])) / (stats.norm.ppf(cp_nc["p_hit"]) - stats.norm.ppf(cp_nc["p_fa"]))

# calculate c' including 'sequence' condition
cp_c = df_c.copy()
cp_c["c_prime"] = -0.5 * (stats.norm.ppf(cp_c["p_hit"]) + stats.norm.ppf(cp_c["p_fa"])) / (stats.norm.ppf(cp_c["p_hit"]) - stats.norm.ppf(cp_c["p_fa"]))

# set up class for testing c'
class TestCprime(unittest.TestCase):

    # test calculation of c' ignoring condition
    def test_cpnc(self):
        cpt_nc = df_nc.copy()
        cpnc_res = sdtt.cprime(cpt_nc, hit_var = "p_hit", fa_var = "p_fa")
        assert_frame_equal(cpnc_res, cp_nc)

    # test calculation of c' including condition
    # NOTE: the test will pass however as this data contains extreme measures the calculation will throw an RuntimeWarning
    def test_cpc(self):
        cpt_c = df_c.copy()
        cpc_res = sdtt.cprime(cpt_c, hit_var = "p_hit", fa_var = "p_fa")
        assert_frame_equal(cpc_res, cp_c)

# --- likelihood ratio (beta)
# calculate beta ignoring 'sequence' condition
b_nc = df_nc.copy()
b_nc["lr_beta"] = -0.5 * (stats.norm.ppf(b_nc["p_hit"])**2 - stats.norm.ppf(b_nc["p_fa"])**2)

# calculate beta including 'sequence' condition
b_c = df_c.copy()
b_c["lr_beta"] = -0.5 * (stats.norm.ppf(b_c["p_hit"])**2 - stats.norm.ppf(b_c["p_fa"])**2)

# set up class for testing likelihood ratio (beta)
class TestLrbeta(unittest.TestCase):

    # test calculation of beta ignoring condition
    def test_bnc(self):
        bt_nc = df_nc.copy()
        bnc_res = sdtt.lrbeta(bt_nc, hit_var = "p_hit", fa_var = "p_fa")
        assert_frame_equal(bnc_res, b_nc)

    # test calculation of beta including condition
    def test_bc(self):
        bt_c = df_c.copy()
        bc_res = sdtt.lrbeta(bt_c, hit_var = "p_hit", fa_var = "p_fa")
        assert_frame_equal(bc_res, b_c)

# TODO: redo each test with corrected data

if __name__ == "__main__":
    unittest.main()
