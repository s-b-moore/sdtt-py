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
dat = pd.read_csv("data/example_data.csv")

print(dat.head())

if __name__ == "__main__":
    unittest.main()

