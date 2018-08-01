import os
import tempfile
import sh
import jinja2
import numpy as np
import StringIO
import xarray as xr
import datetime
import itertools

from multiprocessing.pool import ThreadPool

UVSPEC = sh.Command("/project/meteo/work/Tobias.Koelling/libradtran_git/bin/uvspec")

def run_uvspec(_in, *args, **kwargs):
    try:
        return UVSPEC(_in=_in, _err_to_out=True, *args, **kwargs).stdout
    except Exception as e:
        print "ERROR IN UVSPEC"
        print "input was:"
        print _in
        print "======================="
        print "error was:"
        print e.stdout
        raise RuntimeError("ERROR IN UVSPEC:\n" + _in + "\n" + "="*20 + "\n" + e.stderr)

def gauss(x, mu, sigma):
    sigma2 = sigma**2
    return (2*np.pi*sigma2)**-0.5 * np.exp(-(x-mu)**2 / (2*sigma2))

class NoPool(object):
    def map(self, *args, **kwargs):
        return map(*args, **kwargs)

def _main():
    import sys
    import argparse

    parser = argparse.ArgumentParser("nk lut generator")
    parser.add_argument("datafile")
    parser.add_argument("lutfile")
    parser.add_argument("--srf", default=False, action="store_true",
            help="calculate with spectral response functions")
    args = parser.parse_args()

    env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(os.path.join(os.path.dirname(__file__), 'templates'))
            )
    maintpl = env.get_template("uvspec.tpl.inp")
    wctpl = env.get_template("wc.tpl.inp")

    # wvls = np.array([(750., 2.), (870., 2.),
    #                  (1650., 7.), (1700., 7.), (1750., 7.),
    #                  (2150., 7.), (2200., 7.)])
    ds = xr.open_dataset(args.datafile)
    if args.srf:
        wvls = np.stack([ds.wavelength.data, ds.fwhm.data], axis=1)
    else:
        wvls = np.stack([ds.wavelength.data, np.zeros_like(ds.wavelength.data)], axis=1)
    print "calculating lut for:", wvls

    cloud_base = 1000
    cloud_height = 500

    """
    reffs = [2., 3., 4., 5., 6., 7., 8., 10., 12., 14., 16., 18., 20., 22., 24., 25.]
    lwps = [0., 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1., 2., 3., 4., ]
    szas = [0, 1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 88, 89, 90]
    umus = sorted(np.cos(np.deg2rad(szas)))
    phis = range(0, 181, 15)
    """

    #reffs = [6., 6.5, 7., 7.5, 8., 8.2, 8.4, 8.6, 8.8, 9., 9.2, 9.4, 9.6, 9.8, 10., 10.2, 10.4, 10.6, 10.8, 11., 11.5, 12.]
    #lwps = [0., 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.75, 1., 2., 3., 4., ]
    reffs = np.arange(5., 14., 0.2)
    lwps = np.arange(0., 1., 0.02)
    szas = [0, 1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 88, 89, 90]
    umus = sorted(np.cos(np.deg2rad(szas)))
    phis = range(0, 181, 15)

    table = np.zeros((len(reffs), len(lwps), len(wvls), len(szas), len(umus), len(phis)))

    total_count = np.prod(table.shape[:-2])
    count = [0]
    tstart = datetime.datetime.now()

    p = ThreadPool(20)
    #p = NoPool()

    def do_count():
        count[0] += 1
        t = datetime.datetime.now()
        dt = float((t-tstart).total_seconds())
        tavg = dt/ count[0]
        ttotal = tavg * total_count
        print "collected {:5}/{:5} entries, {:.1f}s elapsed, {:.1f}s to go".format(count[0], total_count, dt, ttotal - dt)

    def calc(((i,reff), (j,lwp), (k,(wvl, fwhm)), (l,sza))):
        dwvl = np.ceil(5.*fwhm)
        wvl_grid_relative = np.arange(-dwvl, dwvl, 0.1)
        sigma = fwhm / (2*(2*np.log(2))**0.5)
        slitfunction = gauss(wvl_grid_relative, 0, sigma)
        slitfunction_data = np.stack([wvl_grid_relative, slitfunction], axis=1)

        with tempfile.NamedTemporaryFile(prefix="/dev/shm/tmp") as wcfile:
            wcdata = wctpl.render(cloud_base=cloud_base,
                                  cloud_height=cloud_height,
                                  lwp=lwp*1000, # uvspec wants g/m2
                                  reff=reff)
            wcfile.write(wcdata)
            wcfile.flush()
            with tempfile.NamedTemporaryFile(prefix="/dev/shm/tmp") as slitfile:
                np.savetxt(slitfile.name, slitfunction_data, fmt="%.10f")


                render_args = dict(umus=umus,
                                   phis=phis,
                                   sza=sza,
                                   center_wavelength = np.round(wvl, decimals=1),
                                   wavelength=wvl,
                                   wcfile=wcfile.name)

                if args.srf:
                    render_args["slitfile"] = slitfile.name
                    render_args["dwvl"] = dwvl
                
                inp = maintpl.render(**render_args)
                output = run_uvspec(_in=inp)
        sio = StringIO.StringIO(output)
        data = np.loadtxt(sio)
        table[i,j,k,l] = data[1:].reshape((len(umus), len(phis)))
        do_count()

    combinations = itertools.product(*map(enumerate, (reffs, lwps, wvls, szas)))
    p.map(calc, combinations)
    #p.map(calc, [combinations.next()])
    ds = xr.Dataset({
        "wvl": xr.DataArray(wvls[:,0], dims=("wvl",)),
        "fwhm": xr.DataArray(wvls[:,1], dims=("wvl",)),
        "reff": xr.DataArray(reffs, dims=("reff",)),
        "lwp": xr.DataArray(lwps, dims=("lwp",)),
        "sza": xr.DataArray(szas, dims=("sza",)),
        "umu": xr.DataArray(umus, dims=("umu",)),
        "phi": xr.DataArray(phis, dims=("phi",)),
        "transmittance": xr.DataArray(table,
            dims=("reff", "lwp", "wvl", "sza", "umu", "phi")),
        })
    ds.to_netcdf(args.lutfile)

if __name__ == "__main__":
    _main()
