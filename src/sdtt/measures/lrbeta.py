# import required libraries
from scipy import stats

# function to calculate likelihood ratio (beta)
def lrbeta(data, hit_var, fa_var):

    dat = data.copy()

    dat["lr_beta"] = -0.5 * (stats.norm.ppf(dat[hit_var])**2 - stats.norm.ppf(dat[fa_var])**2)

    return dat
