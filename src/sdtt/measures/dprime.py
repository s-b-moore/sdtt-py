# import required libraries
from scipy import stats

# function to calculate d'
def dprime(data, hit_var, fa_var):

    dat = data.copy()

    # calculate d'
    dat["d_prime"] = stats.norm.ppf(dat[hit_var] - stats.norm.ppf(dat[fa_var]))

    return dat
