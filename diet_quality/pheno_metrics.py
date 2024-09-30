import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit

def pheno_fq_metrics(ndvi_ts_mean, produce_ts=True, b_start=None, b_end=None):


    """
    ndvi_ts_mean (1-d array): time series of NDVI values for an entire calendar year (e.g., mean for a single pasture)
    produce_ts (boolean): whether to return the entire time series (default) or just average between b_start and b_end (see below)
    b_start (int): the day of year for the start of the time series subset to average over for output. Only used if produce_ts==False.
    b_end (int): the day of the year for the end of the time series subset to average over for output. Only used if produce_ts==False.
    """

    def running_mean(x, N):
        cumsum = np.nancumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    def sigmoid(x, L, x0, k, b):
        y = L / (1 + np.exp(-k * (x - x0))) + b
        return (y)

    def double_logistic(x, vmin, vmax, sos, scaleS, eos, scaleA):
        y = vmin + vmax * ( (1 / (1 + np.exp(-scaleS * (x - sos)))) + (1 / (1 + np.exp(scaleA * (x - eos)))) - 1 )
        return (y)

    def ndvi_int_calc(ts, base, sos):
        ts_tmp = ts - base
        ndvi_int_ts = np.ones_like(ts_tmp) * np.nan
        for b_i in range(ts_tmp.shape[0]):
            ndvi_int_ts[b_i] = np.nansum(ts_tmp[sos:b_i + 1])
        return ndvi_int_ts

    # get length of time series
    b = len(ndvi_ts_mean)
    
    # calculate start of season and base ndvi
    ndvi_thresh1 = np.nanpercentile(ndvi_ts_mean[91:201], 40.0)
    date_thresh1 = next(x for x in np.where(ndvi_ts_mean > ndvi_thresh1)[0] if x > 30)
    dndvi_ts_mean = np.ones_like(ndvi_ts_mean) * np.nan
    dndvi_ts_mean[25:] = running_mean(np.diff(ndvi_ts_mean), 25)
    dndvi_thresh2 = np.nanpercentile(dndvi_ts_mean[:date_thresh1], 35.0)
    sos = np.where(dndvi_ts_mean[:date_thresh1] < dndvi_thresh2)[0][-1]
    ndvi_base = np.nanmean(ndvi_ts_mean[10:75])

    # calculate 'instantaneous greenup rate (IGR)' with potentially different lags
    ndvi_ts_smooth_d1 = np.diff(ndvi_ts_mean, prepend=ndvi_ts_mean[0])

    ndvi_ts_smooth_d1_cum30 = np.empty_like(ndvi_ts_smooth_d1)
    for i in range(b):
        ndvi_ts_smooth_d1_cum30[i] = np.nansum(ndvi_ts_smooth_d1[i - 30:i], axis=0)

    # cleanup and reshape IGR metrics
    ndvi_ts_smooth_d1_cum30[np.where(np.isnan(ndvi_ts_smooth_d1))] = np.nan

    # calculate the peak of IGR 
    window = int(len(ndvi_ts_mean[sos:]) / 2)
    if (window % 2) == 0:
        window -= 1
    ydata = np.copy(ndvi_ts_mean)
    ydata[:sos] = ndvi_base
    xdata = np.arange(len(ydata))
    try:
        p0 = [ndvi_base, np.max(ydata),
              int(np.percentile(xdata, 25)), 1.0, int(np.percentile(xdata, 75)),
              1.0]  # this is a mandatory initial guess
        popt, pcov = curve_fit(double_logistic, xdata, ydata, p0, method='lm', maxfev=20000)
        curvedata = double_logistic(xdata, vmin=popt[0], vmax=popt[1], sos=popt[2],
                                    scaleS=popt[3], eos=popt[4], scaleA=popt[5])
        cum_peak = np.argmax(np.diff(curvedata))
    except RuntimeError:
        try:
            p0 = [ndvi_base, np.max(ydata),
                  sos + 30, 1.0, int(np.percentile(xdata, 75)),
                  1.0]  # this is a mandatory initial guess
            popt, pcov = curve_fit(double_logistic, xdata, ydata, p0, method='lm', maxfev=20000)
            curvedata = double_logistic(xdata, vmin=popt[0], vmax=popt[1], sos=popt[2],
                                        scaleS=popt[3], eos=popt[4], scaleA=popt[5])
            cum_peak = np.argmax(np.diff(curvedata))
        except RuntimeError:
            print(RuntimeError)

    # calculate integrated ndvi
    ndvi_int_ts = ndvi_int_calc(ndvi_ts_mean, ndvi_base, sos)
    ndvi_rate_ts = np.zeros_like(ndvi_int_ts)

    # calculate rate of chagen
    ndvi_rate_ts[sos:] = ndvi_int_ts[sos:] / (range(sos, ndvi_int_ts.shape[0]) - sos + 1)

    # calculate percent dry biomass estimate
    ndvi_dry_ts = np.zeros_like(ndvi_int_ts)
    ndvi_int_dry_ts = np.zeros_like(ndvi_int_ts)
    for i in range(sos, ndvi_dry_ts.shape[0]):
        if ndvi_ts_smooth_d1[i] < 0:
            ndvi_dry_ts[i] = (-1.0 * ndvi_ts_smooth_d1[i] / np.nanmax(ndvi_ts_mean[:i])) * ndvi_int_ts[i]
        ndvi_int_dry_ts[i] = np.nansum(ndvi_dry_ts[:i])

    ndvi_int_dry_pct_ts = np.zeros_like(ndvi_int_ts)
    ndvi_int_dry_pct_ts[ndvi_int_ts != 0] = ndvi_int_dry_ts[ndvi_int_ts != 0] / ndvi_int_ts[ndvi_int_ts != 0]
    
    if produce_ts:
        df_out = pd.DataFrame(
            {
                'NDVI': ndvi_ts_mean,
                'NDVI_d30': ndvi_ts_smooth_d1_cum30,
                't_peak_IRG': np.arange(b) - cum_peak,
                'iNDVI':ndvi_int_ts,
                'iNDVI_dry': ndvi_int_dry_ts,
                'NDVI_rate': ndvi_rate_ts,
                'iNDVI_dry_pct': ndvi_int_dry_pct_ts,
                'SOS_doy': sos
            }
        )
        return df_out
    elif (b_start is not None) and (b_end is not None):    
        # get days since peak IRG
        days_cum_peak = (b_end - 1) - cum_peak
    
        # get final outputs at pasture scale
        ndvi = np.nanmean(ndvi_ts_mean[b_start:b_end])
        ndvi_d1_cum30 = np.nanmean(ndvi_ts_smooth_d1_cum30[b_start:b_end])
        ndvi_int = np.nanmean(ndvi_int_ts[b_start:b_end])
        ndvi_int_dry = np.nanmean(ndvi_int_dry_ts[b_start:b_end])
        ndvi_rate = np.nanmean(ndvi_rate_ts[b_start:b_end])
        ndvi_int_dry_pct = np.nanmean(ndvi_int_dry_pct_ts[b_start:b_end])
        
        return {
            'NDVI': ndvi.astype('float32'),
            'NDVI_d30': ndvi_d1_cum30.astype('float32'),
            't_peak_IRG': days_cum_peak.astype('int'),
            'iNDVI': ndvi_int.astype('float32'),
            'iNDVI_dry': ndvi_int_dry.astype('float32'),
            'NDVI_rate': ndvi_rate.astype('float32'),
            'iNDVI_dry_pct': ndvi_int_dry_pct.astype('float32'),
            'SOS_doy': sos
        }
    else:
        print('ERROR: Set to output single value (produce_ts=False) but b_start and/or b_end are None. This is not allowed.')