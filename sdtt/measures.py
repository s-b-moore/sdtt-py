# import required libraries
from scipy import stats

# function to calculate d'
def dprime(data, hit_var, fa_var):
    dat = data.copy()
    dat["d_prime"] = stats.norm.ppf(dat[hit_var]) - stats.norm.ppf(dat[fa_var])
    return dat

# function to calculate criterion location (c)
def criterion(data, hit_var, fa_var):
    dat = data.copy()
    dat["criterion"] = stats.norm.ppf(dat[hit_var] + stats.norm.ppf(dat[fa_var]))
    return dat

# TODO: c' calculation currently not functional (need to implement checker for extreme values)
# function to calculate c'
def cprime(data, hit_var, fa_var):
    dat = data.copy()
    dat["c_prime"] = -0.5 * (stats.norm.ppf(dat[hit_var]) + stats.norm.ppf(dat[fa_var])) / (stats.norm.ppf(dat[hit_var]) - stats.norm.ppf(dat[fa_var]))
    return dat

# function to calculate likelihood ratio (beta)
def lrbeta(data, hit_var, fa_var):
    dat = data.copy()
    dat["lr_beta"] = -0.5 * (stats.norm.ppf(dat[hit_var])**2 - stats.norm.ppf(dat[fa_var])**2)
    return dat
