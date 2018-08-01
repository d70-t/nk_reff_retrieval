# -*- encoding: utf-8 -*-
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

class PixelwiseLut(object):
    def __init__(self, lut, ppl, angles=None):
        self.lut = lut
        self.ppl = ppl
        self.angles = angles

    def retrieve(self, measurement):
        delta = self.ppl - measurement.transmission.data[...,np.newaxis,np.newaxis]
        distance = np.linalg.norm(delta, axis=2)
        valid = np.isfinite(distance.sum(axis=(-2,-1)))
        valid &= measurement.shadow_flag.data == 0
        distance[~valid] = 1e90
        df = distance.reshape(distance.shape[:2] + (-1,))
        #plt.figure()
        #plt.imshow(distance[1300, 120])
        #plt.colorbar()
        #plt.title("distance")
        #plt.savefig("distance.png")
        #plt.close()
        idxf = np.argmin(df, axis=-1)
        idx = np.unravel_index(idxf, distance[0,0].shape)
        #plt.figure()
        #plt.imshow(idx[0])
        #plt.title("reff indices")
        #plt.savefig("reff_indices.png")
        #plt.close()
        reff = self.lut.lut.reff.data[idx[0]].astype("float")
        reff[~valid] = np.nan
        lwp = self.lut.lut.lwp.data[idx[1]].astype("float")
        lwp[~valid] = np.nan
        return xr.Dataset({
            "reff": xr.DataArray(reff, dims=measurement.transmission.dims[:2]),
            "lwp": xr.DataArray(lwp, dims=measurement.transmission.dims[:2]),
            })

    def __getitem__(self, item):
        return PixelwiseLut(self.lut,
                            self.ppl[item],
                            angles[item] if angles is not None else None)

    def plot(self, ax):
        assert len(self.ppl.shape) == 3, "only single pixel ppls can be plotted"
        ax.set_title(u"LUT sza: {:.1f}° vza: {:.1f}° dphi: {:.1f}°".format(
                     np.rad2deg(self.angles.sza),
                     np.rad2deg(self.angles.vza),
                     np.rad2deg(self.angles.daz)))

        for i in range(self.ppl.shape[1]):
            ax.plot(self.ppl[0,i], self.ppl[1,i], color="black", alpha=.3)
        for i in range(self.ppl.shape[2]):
            ax.plot(self.ppl[0,:,i], self.ppl[1,:,i], color="black", alpha=.3)

        import matplotlib.cm as cm
        for color, values, reff in zip(cm.viridis(np.linspace(0, 1, self.ppl.shape[1])), self.ppl.transpose(1,0,2), self.lut.lut.reff.data):
            ax.scatter(values[0], values[1], color=color, label="reff = {:.0f}um".format(reff))
        ax.set_xlabel("normalized radiance {:.1f} nm".format(self.lut.lut.wvl.data[0]))
        ax.set_ylabel("normalized radiance {:.1f} nm".format(self.lut.lut.wvl.data[1]))
        ax.set_xlim(0, 0.3)
        ax.set_ylim(0, 0.2)


class NKLut(object):
    def __init__(self, lut):
        """
        :param lut: Lookoup Table as xarray dataset
        """
        assert lut.transmittance.dims[0] == "reff"
        assert lut.transmittance.dims[1] == "lwp"
        assert lut.transmittance.dims[2] == "wvl"
        assert lut.transmittance.dims[3] == "sza"
        assert lut.transmittance.dims[4] == "umu"
        assert lut.transmittance.dims[5] == "phi"
        self.lut = lut
        self.interpolator = RegularGridInterpolator((lut.sza, lut.umu, lut.phi),
                                                    lut.transmittance.data.transpose(3,4,5,2,0,1),
                                                    bounds_error=False)
    def create_per_pixel_luts(self, angles):
        return PixelwiseLut(self, self.interpolator((np.rad2deg(angles.sza), np.cos(angles.vza), np.rad2deg(angles.daz))), angles)

class Angles(object):
    def __init__(self, sza, vza, daz):
        self.sza = sza
        self.vza = vza
        self.daz = daz
