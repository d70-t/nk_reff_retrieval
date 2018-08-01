# -*- encoding: utf-8 -*-
import itertools

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from plot_angles import calc_daz
from lut_tools import PixelwiseLut, NKLut

class FlatCloudAngles(object):
    """
    Provider for measurement angles considering a flat cloud.
    """
    def __init__(self, ds):
        """
        :param ds: measurement dataset
        """
        if "frames" not in ds.sun_rays.dims:
            sun_rays = ds.sun_rays.data * np.ones((ds.dims["frames"], 1))
        else:
            sun_rays = ds.sun_rays.data
        if "frames" not in ds.zenith.dims:
            zenith = ds.zenith.data * np.ones((ds.dims["frames"], 1))
        else:
            zenith = ds.zenith.data
        self.vza = np.arccos(np.einsum("ik,ijk->ij", zenith, ds.outgoing_rays))
        self.sza = np.arccos(np.einsum("ik,ik->i", zenith, sun_rays))[:,np.newaxis] * np.ones(ds.dims["spatial"])
        self.daz = calc_daz(zenith[:,np.newaxis,:], sun_rays[:,np.newaxis,:], -ds.outgoing_rays)
        print self.vza.shape
        print self.sza.shape
        print self.daz.shape

class StructuredCloudAngles(object):
    """
    Provider for measurement angles considering a 3d structured cloud.
    """
    def __init__(self, ds):
        """
        :param ds: measurement dataset
        """
        self.vza = np.arccos(np.abs(np.einsum("ijk,ijk->ij", ds.surface_normals, ds.outgoing_rays)))
        if "frames" in ds.sun_rays.dims:
            self.sza = np.arccos(np.abs(np.einsum("ijk,ik->ij", ds.surface_normals, ds.sun_rays)))
            self.daz = calc_daz(ds.surface_normals, ds.sun_rays.data[:,np.newaxis,:], -ds.outgoing_rays)
        else:
            self.sza = np.arccos(np.abs(np.einsum("ijk,k->ij", ds.surface_normals, ds.sun_rays)))
            self.daz = calc_daz(ds.surface_normals, ds.sun_rays.data[np.newaxis, np.newaxis,:], -ds.outgoing_rays)


def show_pixel_lut(x, y, ppl, measurement):
    l = ppl.ppl[x,y]
    m = measurement.transmission[x,y]
    if ppl.angles.sza.shape == 1:
        sza = ppl.angles.sza[x]
    elif ppl.angles.sza.shape[1] == 1:
        sza = ppl.angles.sza[x, 0]
    else:
        sza = ppl.angles.sza[x,y]
    plt.figure()
    plt.title(u"LUT of pixel {} {}, sza: {:.1f}° vza: {:.1f}° dphi: {:.1f}°".format(x, y,
                    np.rad2deg(sza),
                    np.rad2deg(ppl.angles.vza[x,y]),
                    np.rad2deg(ppl.angles.daz[x,y])))
    for i in range(l.shape[1]):
        plt.plot(l[0,i], l[1,i], color="black", alpha=.3)
    for i in range(l.shape[2]):
        plt.plot(l[0,:,i], l[1,:,i], color="black", alpha=.3)

    import matplotlib.cm as cm
    for color, values, reff in zip(cm.viridis(np.linspace(0, 1, l.shape[1])), l.transpose(1,0,2), ppl.lut.lut.reff.data):
        plt.scatter(values[0], values[1], color=color, label="reff = {:.0f}um".format(reff))
    plt.scatter(m[0], m[1], color="red", marker="x", label="measurement")
    plt.xlabel("transmission {:.1f} nm".format(measurement.wavelength.data[0]))
    plt.ylabel("transmission {:.1f} nm".format(measurement.wavelength.data[1]))
    plt.xlim(0, 0.3)
    plt.ylim(0, 0.2)
    plt.legend(loc=1)
    plt.savefig("ppls/per_pixel_lut_{}_{}.png".format(x, y))
    plt.close()

def _main():
    import argparse
    parser = argparse.ArgumentParser("reff retrieval")
    parser.add_argument("measurement", type=str, help="netCDF file conatining measurement data")
    parser.add_argument("lut", type=str, help="netCDF file containing lookup tables")
    parser.add_argument("out", type=str, help="netCDF file for output")
    parser.add_argument("-s", "--structured", default=False, action='store_true', help="use cloud structure for retrieval")

    args = parser.parse_args()

    measurement = xr.open_dataset(args.measurement)
    lut = NKLut(xr.open_dataset(args.lut).sel(reff=slice(4,None)))
    if args.structured:
        angles = StructuredCloudAngles(measurement)
    else:
        angles = FlatCloudAngles(measurement)

    ppl = lut.create_per_pixel_luts(angles)

    #if True:
    if False:
        #for x in [1300, 1350, 1400, 1450]:
        #    for y in [80, 90, 100, 110, 120]:
        for x in range(20): ##range(0, measurement.dims["frames"], 10):
            for y in [10]: #range(0, measurement.dims["spatial"], 50):
                show_pixel_lut(x, y, ppl, measurement)
    res = ppl.retrieve(measurement)
    res["cth"] = measurement.cth
    res["radiance"] = measurement.radiance
    res["shadow_flag"] = measurement.shadow_flag
    res["sza"] = xr.DataArray(angles.sza, dims=res.cth.dims)
    res["vza"] = xr.DataArray(angles.vza, dims=res.cth.dims)
    res["daz"] = xr.DataArray(angles.daz, dims=res.cth.dims)
    res.to_netcdf(args.out)

if __name__ == "__main__":
    _main()
