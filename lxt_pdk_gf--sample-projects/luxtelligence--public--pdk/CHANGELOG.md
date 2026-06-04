# CHANGELOG

The release naming convention follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v1.3.0

### New


- Added high-speed versions of mzm_unbalanced and eo_phase_shifter #107
- Making project compatible with gdsfactoryplus #105
- Support 2x2 MMI in MZM cell

### Bug fixes

- Fix layerspec import
- MZM caching bugfix


## v1.2.0

### New

- Balanced directional coupler building block
- Add wavelength dependence to phase shifter model
- Add thermo-optical phase shifter cell

## v0.1.1

### Bug fixes
- Fix typo for the heater layer in LayerStack
- Fix optics layer names (ridge, slab)
- Use center parameter in chip_frame

## v0.1.0

### New
- Technology definition for lnoi400
- Basic Component definitions: bends, edge coupler, mmis, uniform CPWs, phase and amplitude modulators
- Circuit models based on [sax](https://github.com/flaport/sax)
- Example notebooks for:
    - Layout with routing to the chip facets
    - Circuit simulation
