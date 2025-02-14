# import required libraries
import sys
import pandas as pd
import unittest

# set the path to import the sdtt module
par_dir = ".."
sys.path.append(par_dir)

# import sdtt module
import sdtt

# generate data for testing
dat_hit = pd.DataFrame({"accuracy": [1], "signal": [1]})
dat_miss = pd.DataFrame({"accuracy": [0], "signal": [1]})
dat_cr = pd.DataFrame({"accuracy": [1], "signal": [0]})
dat_fa = pd.DataFrame({"accuracy": [0], "signal": [0]})

# set up class for testing
class TestOut(unittest.TestCase):

    def test_hit(self):
        hit_res = sdtt.outcomes(dat_hit, "accuracy", "signal")
        self.assertEqual(hit_res.loc[0, "outcome"], "hit")

    def test_miss(self):
        miss_res = sdtt.outcomes(dat_miss, "accuracy", "signal")
        self.assertEqual(miss_res.loc[0, "outcome"], "miss")

    def test_cr(self):
        cr_res = sdtt.outcomes(dat_cr, "accuracy", "signal")
        self.assertEqual(cr_res.loc[0, "outcome"], "correct_rejection")

    def test_fa(self):
        fa_res = sdtt.outcomes(dat_fa, "accuracy", "signal")
        self.assertEqual(fa_res.loc[0, "outcome"], "false_alarm")

if __name__ == "__main__":
    unittest.main()
