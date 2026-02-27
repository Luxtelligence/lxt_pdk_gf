# Luxtelligence Process Design Kit (PDK) for gdsfactory

![Luxtelligence](LXT_Logo.png)

[Luxtelligence](https://luxtelligence.ai/) Process Design Kit (PDK) for gdsfactory. The Luxtelligence PDK contains a library of components that facilitate the design of photonic integrated circuits for Luxtelligence's foundry service. The PDK includes both electrical and optical building blocks that leverage Lithium Tantalate and Lithium Niobate's electro-optic effect and attractive optical properties. Each building block consists of a geometrical layout, defining the starting point for microfabrication of the integrated circuit, and a compact circuit model that approximates the real frequency-domain behaviour of the component.

The `lxt_pdk_gf` PDK is released open-source to allow users to easily evaluate a sample of what Luxtelligence has to offer. Please [contact us](mailto:foundry@luxtelligence.ai) for information on advanced building blocks and variations on the standard PDK geometry.

## Installation

Python 3.12 is required. We recommend [VSCode](https://code.visualstudio.com/) or [Google Antigravity](https://antigravity.google/) as IDEs.

If you do not have Python installed, you can [download Anaconda](https://www.anaconda.com/download/).Once Python is available, clone the repository and install the package in editable mode:

```
git clone https://github.com/Luxtelligence/lxt_pdk_gf.git
cd lxt_pdk_gf
pip install -e . pre-commit
pre-commit install
python install_tech.py
```

Restart KLayout afterwards to ensure the newly installed technology appears.

## Examples

After installing the PDK, you can verify that it is working correctly by running the Jupyter notebooks in the [docs/notebooks](https://github.com/Luxtelligence/lxt_pdk_gf/tree/main/docs/notebooks) folder.

## Documentation

- [PDK documentation](https://luxtelligence.github.io/lxt_pdk_gf/)
- [gdsfactory documentation](https://gdsfactory.github.io/gdsfactory/)
