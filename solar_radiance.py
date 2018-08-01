import datetime
import calendar
import numpy as np
import xarray as xr

def day_angle(day):
    """
    Angle of the earth's motion around the sun.
    See Iqbal (1983), page 3.

    :param day: Either day of year (1 == Jan 1st), without counting the leap day
                Or a datetime object.
    """
    try:
        tt = day.timetuple()
    except AttributeError:
        pass
    else:
        is_leap = calendar.isleap(tt.tm_year)
        day = tt.tm_yday
        if day >= 60 and is_leap:
            day -= 1
    return 2 * np.pi * (day - 1) / 365.

def excentricity_correction(day):
    """
    Correction for extraterrestrial flux (E0 == (r0/r)**2).
    This factor can be multiplied with the extraterrestrial flux
    at a distance of 1 AU to compensate for the annual variation.
    See Iqbal (1983), page 3.

    :param day: Either day of year (1 == Jan 1st), without counting the leap day
                Or a datetime object.
    """
    angle = day_angle(day)
    return 1.000110 + \
           0.034221 * np.cos(angle) + \
           0.001280 * np.sin(angle) + \
           0.000719 * np.cos(2*angle) + \
           0.000077 * np.sin(2*angle)

def load_solar_spectrum(filename):
    """
    Load solar spectrum data according to libradtran convention.
    """
    data = np.loadtxt(filename)
    return xr.Dataset({
        "wavelength": xr.DataArray(data[:,0],
                                   dims=("wavelength",),
                                   attrs={"units": "nm"}),
        "flux": xr.DataArray(data[:,1],
                             dims=("wavelength",),
                             attrs={"units": "mW m-2 nm-1"})
        })

def gauss(x, mu, sigma):
    sigma2 = sigma**2
    return (2*np.pi*sigma2)**-0.5 * np.exp(-(x-mu)**2 / (2*sigma2))

def resample_spectrum_gaussian(spectrum, sensor_parameters):
    wvl = spectrum.wavelength.data
    dwvl = wvl[1:] - wvl[:-1]
    spectral_weight = np.empty_like(wvl)
    spectral_weight[1:-1] = 0.5 * (dwvl[:1] + dwvl[1:])
    spectral_weight[0] = dwvl[0]
    spectral_weight[-1] = dwvl[-1]

    sensor_weight = gauss(wvl[np.newaxis,:],
                          sensor_parameters.center_wavelength.data[:,np.newaxis],
                          sensor_parameters.fwhm.data[:,np.newaxis]/(2*(2*np.log(2))**.5))
    
    weight = spectral_weight[np.newaxis,:] * sensor_weight
    weight /= np.sum(weight, axis=-1)[:,np.newaxis]

    resampled = np.einsum("ij,j->i", weight, spectrum.flux.data)

    return xr.Dataset({
        "wavelength": xr.DataArray(sensor_parameters.center_wavelength.data,
                                   dims="wavelength"),
        "flux": xr.DataArray(resampled, dims="wavelength")
        })

