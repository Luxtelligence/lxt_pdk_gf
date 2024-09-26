We provide circuit models for the PDK elements, implemented using the [sax](https://flaport.github.io/sax/) circuit simulator. A dictionary with the available models can be obtained by running::

  models = lnoi400.get_models()

These models are useful for constructing the scattering matrix of a building block, or of a hierarchical circuit. They are obtained by experimental characterization of the building blocks and with FDTD simulations,
that provide the full wavelength-dependent behaviour. Since the lnoi400 PDK is conceived for optical C-band operation, the model results should not be trusted below 1500 or above 1600 nm.

Examples of circuit simulation using sax
---------------------------------------------------------------------------

.. nbinput:: ../lnoi400/examples/circuit_simulation.ipynb
