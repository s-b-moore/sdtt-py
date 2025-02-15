# import required libraries
from scipy import stats

# function to calculate criterion location (c)
def criterion(data, hit_var, fa_var):

    dat = data.copy()

    # calculate criterion location (c)
    dat["criterion"] = -0.5 * (stats.norm.ppf(dat[hit_var]) + stats.norm.ppf(dat[fa_var]))

    return dat
