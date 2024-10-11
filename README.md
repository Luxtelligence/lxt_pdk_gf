# lnoi400

![Luxtelligence](LXT_Logo.png)

[Luxtelligence](https://luxtelligence.ai/) lnoi400 Process Design Kit (PDK). The Luxtelligence Process Design Kit contains a library of components that facilitate the design of photonic integrated circuits for Luxtelligence's Lithium Niobate on Insulator (LNOI) foundry service. The PDK includes both electrical and optical building blocks, that make use of Lithium Niobate's electro-optic effect and attractive optical properties. PDK building blocks consist both of a geometrical layout, defining the starting point for the microfabrication of the integrated circuit, and a circuit model, that approximates the real frequency-domain behaviour of the optical component.

The lnoi400 PDK is released open-source, to let users easily evaluate a sample of the offering of Luxtelligence. Please [contact us](mailto:foundry@luxtelligence.ai) for information on advanced building blocks and variations on the standard PDK geometry.

## Installation

Use python3.11 or python3.12. We recommend [VSCode](https://code.visualstudio.com/) as an IDE.

If you don't have python installed on your system you can [download anaconda](https://www.anaconda.com/download/)

Once you have python installed, open Anaconda Prompt as Administrator and then install the latest gdsfactory using pip.

![anaconda prompt](https://i.imgur.com/eKk2bbs.png)


```
git clone https://github.com/Luxtelligence/lxt_pdk_gf.git
cd lxt_pdk_gf
pip install -e . pre-commit
pre-commit install
python install_tech.py
```
Then you need to restart Klayout to make sure the new technology installed appears.

## Examples

After installing the PDK in your environment, you can validate that it is correctly working by running the Jupyter notebooks in the examples folder [lnoi400/examples](https://github.com/Luxtelligence/lxt_pdk_gf/tree/main/lnoi400/examples).

## Documentation

- [gdsfactory docs](https://gdsfactory.github.io/gdsfactory/)
