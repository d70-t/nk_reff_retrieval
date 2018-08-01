import os
import xarray as xr
import runmacs.spec.calibration.calibrationdata as cd
import yaml

DATAPATH = os.path.join(os.path.dirname(__file__), "..", "data")
CALFILES = {"VNIR": os.path.join(DATAPATH, "specMACS_VNIR_cal_CHB+SPECIM_2014_v1_plain_nawdex"),
            "SWIR": os.path.join(DATAPATH, "specMACS_SWIR_cal_CHB+SPECIM_2016_NARVAL_NAWDEX_temp")}

def sensor_params(sensor):
    cal = cd.CalibrationData.fromFile(CALFILES[sensor])
    spectral = cal.wvlns.shape[0]/2
    return xr.Dataset({
        "center_wavelength": xr.DataArray(cal.wvlns[spectral], dims=("wavelength",)),
        "fwhm": xr.DataArray(cal.wvlns_fwhm[spectral], dims=("wavelength",))})


def _main():
    import sys
    conf = yaml.load(open(sys.argv[1]))
    outfilename = sys.argv[2]

    for sensor, info in conf["specmacs"].items():
        sensor_params(sensor)
