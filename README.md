# Usage

The scripts offer some help with the `-h` command line options.

* `run_uvspec.py` is the tool to calculate lookup tables.
* `retrieve.py` is to run the retrieval.

Both scripts need measurement data in the following netCDF data layout:

## Typical measurement data layout:

```
netcdf surface_measurements_nawdex_flat_combined {
dimensions:
	wavelength = 6 ;
	frames = 3420 ;
	spatial = 320 ;
	nv = 3 ;
variables:
	double wavelength(wavelength) ;
		wavelength:_FillValue = NaN ;
	double distance(frames, spatial) ;
		distance:_FillValue = NaN ;
	float radiance(frames, spatial, wavelength) ;
		radiance:_FillValue = NaNf ;
	double zenith(frames, nv) ;
		zenith:_FillValue = NaN ;
	double cth(frames, spatial) ;
		cth:_FillValue = NaN ;
	double surface_normals(frames, spatial, nv) ;
		surface_normals:_FillValue = NaN ;
	double excentricity_correction ;
		excentricity_correction:_FillValue = NaN ;
	double points(frames, spatial, nv) ;
		points:_FillValue = NaN ;
	double sun_rays(frames, nv) ;
		sun_rays:_FillValue = NaN ;
	double outgoing_rays(frames, spatial, nv) ;
		outgoing_rays:_FillValue = NaN ;
	double fwhm(wavelength) ;
		fwhm:_FillValue = NaN ;
	double solar_flux(wavelength) ;
		solar_flux:_FillValue = NaN ;
	double transmission(frames, spatial, wavelength) ;
		transmission:_FillValue = NaN ;
```
