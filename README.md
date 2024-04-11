# lnoi400 0.0.1

luxtelligence lnoi400 pdk

## Installation

### Installation for users

Use python3.10 or python3.11. We recommend [VSCode](https://code.visualstudio.com/) as an IDE.

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

## Documentation

- [gdsfactory docs](https://gdsfactory.github.io/gdsfactory/)
