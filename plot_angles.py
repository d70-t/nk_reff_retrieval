import numpy as np
import xarray as xr
import matplotlib.pylab as plt

def plot_real_vs_effective(name, real, eff):
    fig, ax = plt.subplots(2)
    lo = min(np.percentile(real, 2), np.percentile(eff, 2))
    hi = max(np.percentile(real, 98), np.percentile(eff, 98))
    for i, (kind, var) in enumerate([("real", real), ("effective", eff)]):
        ax[i].set_title("{} {}".format(kind, name))
        mappable = ax[i].imshow(var, aspect="auto", vmin=lo, vmax=hi)
        plt.colorbar(mappable, ax=ax[i])
    plt.tight_layout()

def calc_daz(normal, a, b):
    """
    :note: daz is given in uvspec convention, meaning:
           * 0deg -> vectors a and b are pointing away from each other
           * 180deg -> vectors a and b are pointing in similar directions
    """
    print normal.shape, a.shape, b.shape
    a_flat = a - np.einsum("...i,...i->...", a, normal)[...,np.newaxis] * normal
    b_flat = b - np.einsum("...i,...i->...", b, normal)[...,np.newaxis] * normal
    return np.arccos(-np.einsum("...i,...i->...", a_flat, b_flat))

def _main():
    import sys
    filename = sys.argv[1]

    ds = xr.open_dataset(filename)
    sza = np.arccos(np.einsum("ij,ij->i", ds.zenith, ds.sun_rays))
    sza_eff = np.arccos(np.abs(np.einsum("ikj,ij->ik", ds.surface_normals, ds.sun_rays)))
    vza = np.arccos(np.einsum("ij,ikj->ik", ds.zenith, ds.outgoing_rays))
    vza_eff = np.arccos(np.abs(np.einsum("ikj,ikj->ik", ds.surface_normals, ds.outgoing_rays)))

    daz = calc_daz(ds.zenith.data[:, np.newaxis, :],
                   ds.sun_rays.data[:, np.newaxis, :],
                   -ds.outgoing_rays)
    daz_eff = calc_daz(ds.surface_normals,
                       ds.sun_rays.data[:, np.newaxis, :],
                       -ds.outgoing_rays)

    plot_real_vs_effective("viewing zenith angle",
                           np.rad2deg(vza.T),
                           np.rad2deg(vza_eff.T))
    plot_real_vs_effective("solar zenith angle",
                           np.rad2deg(sza*np.ones_like(sza_eff.T)),
                           np.rad2deg(sza_eff.T))
    plot_real_vs_effective("differential azimuth angle",
                           np.rad2deg(daz.T),
                           np.rad2deg(daz_eff.T))

    plt.figure()
    ds.radiance.isel(wavelength=0).T.plot(cmap="gray")

    plt.show()

if __name__ == '__main__':
    _main()
