# import required libraries
from scipy import stats

# function to calculate c'
def cprime(data, hit_var, fa_var):

    dat = data.copy()

    dat["c_prime"] = -0.5 * (stats.norm.ppf(dat[hit_var]) + stats.norm.ppf(dat[fa_var])) / (stats.norm.ppf(dat[hit_var]) - stats.norm.ppf(dat[fa_var]))

    return dat
