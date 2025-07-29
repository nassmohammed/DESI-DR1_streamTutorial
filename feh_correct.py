import numpy as np


def betw(x, x1, x2):
    return (x >= x1) & (x < x2)


def calibrate(feh_orig, teff, logg, pipeline='RVS', release='DR1'):
    """
    Take the input feh, teff and logg 
    and return the calibrated feh
    
    Make sure to provide which pipeline it is RVS or SP
    """
    assert release == 'DR1'
    logteff_scale = 0.1
    teff_ref = 5000
    bounds = {
        'RVS': [[1.8, 5.5], [4200, 6600]],
        'SP': [[0, 5.1], [4000, 6600]]
    }

    cuts = {'RVS': 4.3, 'SP': 3.5}
    coeffs_low = {'RVS': [0.197, -0.809, 0.441], 'SP': [0.113, 0.215, -0.146]} # giants
    coeffs_high = {
        'RVS': [0.039, -0.284, 0.079],
        'SP': [0.116, -0.216, -0.081] #dwarves
    }

    coeff_low = coeffs_low[pipeline]
    coeff_high = coeffs_high[pipeline]
    cut = cuts[pipeline]
    bound = bounds[pipeline]
    if pipeline not in ["RVS", "SP"]:
        raise RuntimeError('oops, unknown pipeline')

    subset = betw(logg, bound[0][0], bound[0][1]) & betw(
        teff, bound[1][0], bound[1][1])
    ret = feh_orig * 0 + np.nan
    subset_low = subset & (logg < cut)
    subset_high = subset & (logg >= cut)
    for cur_sub, cur_coeff in ([subset_low,
                                coeff_low], [subset_high, coeff_high]):
        ret[cur_sub] = (feh_orig[cur_sub] - np.poly1d(cur_coeff[::-1])(
            np.log10(teff[cur_sub] / teff_ref) / logteff_scale))
    return ret


def delta(feh_orig, teff, logg, pipeline='RVS', release='DR1'):
    """
    Take the input feh, teff and logg 
    and return the difference between DESI and Calib FEH
    
    Make sure to provide which pipeline it is RVS or SP
    """
    assert release == 'DR1'
    logteff_scale = 0.1
    teff_ref = 5000
    bounds = {
        'RVS': [[1.8, 5.5], [4200, 6600]],
        'SP': [[0, 5.1], [4000, 6600]]
    }

    cuts = {'RVS': 4.3, 'SP': 3.5}
    coeffs_low = {'RVS': [0.197, -0.809, 0.441], 'SP': [0.113, 0.215, -0.146]}
    coeffs_high = {
        'RVS': [0.039, -0.284, 0.079],
        'SP': [0.116, -0.216, -0.081]
    }

    coeff_low = coeffs_low[pipeline]
    coeff_high = coeffs_high[pipeline]
    cut = cuts[pipeline]
    bound = bounds[pipeline]
    if pipeline not in ["RVS", "SP"]:
        raise RuntimeError('oops, unknown pipeline')

    subset = betw(logg, bound[0][0], bound[0][1]) & betw(
        teff, bound[1][0], bound[1][1])
    ret = feh_orig * 0 + np.nan
    subset_low = subset & (logg < cut)
    subset_high = subset & (logg >= cut)
    for cur_sub, cur_coeff in ([subset_low,
                                coeff_low], [subset_high, coeff_high]):
        ret[cur_sub] = (-np.poly1d(cur_coeff[::-1])(
            np.log10(teff[cur_sub] / teff_ref) / logteff_scale))
    return ret

def get_x(feh_orig, teff, logg, pipeline='RVS', release='DR1'):
    """
    Take the input feh, teff and logg 
    and return the difference between DESI and Calib FEH
    
    Make sure to provide which pipeline it is RVS or SP
    """
    assert release == 'DR1'
    logteff_scale = 0.1
    teff_ref = 5000
    bounds = {
        'RVS': [[1.8, 5.5], [4200, 6600]],
        'SP': [[0, 5.1], [4000, 6600]]
    }

    cuts = {'RVS': 4.3, 'SP': 3.5}
    coeffs_low = {'RVS': [0.197, -0.809, 0.441], 'SP': [0.113, 0.215, -0.146]}
    coeffs_high = {
        'RVS': [0.039, -0.284, 0.079],
        'SP': [0.116, -0.216, -0.081]
    }

    coeff_low = coeffs_low[pipeline]
    coeff_high = coeffs_high[pipeline]
    cut = cuts[pipeline]
    bound = bounds[pipeline]
    if pipeline not in ["RVS", "SP"]:
        raise RuntimeError('oops, unknown pipeline')

    subset = betw(logg, bound[0][0], bound[0][1]) & betw(
        teff, bound[1][0], bound[1][1])
    ret = feh_orig * 0 + np.nan
    subset_low = subset & (logg < cut)
    subset_high = subset & (logg >= cut)
    for cur_sub, cur_coeff in ([subset_low,
                                coeff_low], [subset_high, coeff_high]):
        ret[cur_sub] = (np.log10(teff[cur_sub] / teff_ref) / logteff_scale)
    return ret
