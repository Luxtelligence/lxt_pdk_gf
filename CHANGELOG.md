# CHANGELOG

The release naming convention follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v2.1.1

GDSFactory-maintained release based on Luxtelligence v2.1.0, with the additions and fixes below.

### New

- Add `lnoi400.mzm_with_pads` with GSG pads and three-conductor transmission-line tapers.
- Expose individual signal and ground ports on CPW pads and MZMs while preserving legacy MZM ports.

### Bug fixes

- Fix ring-resonator slab generation so shared cached components are not modified.
- Align chip-frame and PHIX die edge-coupler placement to the physical chip edge on layer 6/1. The default straight tip extends 5 µm outside and at least 5 µm inside the chip.
- Document the approximately 30 µm straight section for polishing and the insertion-loss tradeoff for unpolished tips.
- Correct Manhattan port-orientation checks at angle wraparound boundaries.

## v2.0

### New

- Technology definition for ltoi300
- Basic Component definitions: edge couplers, mmis, ring cavities, phase and amplitude modulators for O-band and C-band
- Added first compact models for ltoi300
- Added one more circuit placement example notebook for the ltoi300 technology
- Expand README with KLayout lyp and DRC instructions

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
