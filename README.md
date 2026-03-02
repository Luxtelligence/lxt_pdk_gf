# Luxtelligence Process Design Kit (PDK) for gdsfactory

![Luxtelligence](LXT_Logo.png)

[Luxtelligence](https://luxtelligence.ai/) Process Design Kit (PDK) for gdsfactory. The Luxtelligence PDK contains a library of components that facilitate the design of photonic integrated circuits for Luxtelligence's foundry service. The PDK includes both electrical and optical building blocks that leverage Lithium Tantalate and Lithium Niobate's electro-optic effect and attractive optical properties. Each building block consists of a geometrical layout, defining the starting point for microfabrication of the integrated circuit, and a compact circuit model that approximates the real frequency-domain behaviour of the component.

The `lxt_pdk_gf` PDK is released open-source to allow users to easily evaluate a sample of what Luxtelligence has to offer. Please [contact us](mailto:foundry@luxtelligence.ai) for information on advanced building blocks and variations on the standard PDK geometry.

## Installation

Python 3.12 is required. We recommend [VSCode](https://code.visualstudio.com/) or [Google Antigravity](https://antigravity.google/) as IDEs.

If you do not have Python installed, you can [download Anaconda](https://www.anaconda.com/download/). Once Python is available, clone the repository and install the package in editable mode:

```
git clone https://github.com/Luxtelligence/lxt_pdk_gf.git
cd lxt_pdk_gf
pip install -e .
python install_tech.py
```

Restart KLayout afterwards to ensure the newly installed technology appears.

## KLayout Layer Properties

Each PDK has a `klayout/` folder containing `.lyp` layer property files (e.g. `ltoi300/klayout/` and `lnoi400/klayout/`). These files define the colours, fill patterns, and display names for every process layer.

To activate them in KLayout:

1. Open KLayout and go to **File → Setup**.
2. Navigate to the **Application** section and select **Layer Properties**.
3. Under **Default layer properties file**, click **Browse** and point it to the `.lyp` file for your PDK (e.g. `ltoi300/klayout/ltoi300.lyp`).
4. Click **Apply** and **OK**. Restart KLayout to apply the changes.

## KLayout DRC

Design Rule Check (DRC) scripts for each process stack can be downloaded from [luxtelligence.ai](https://luxtelligence.ai/). The files have a `.lydrc` extension and are specific to the technology stack you are using.

**Installation:**

Place the downloaded `.lydrc` file(s) in your local KLayout DRC folder:

```
<KLayout user folder>/drc/
```

> **Note:** KLayout has a known issue where only the first DRC file in the `drc/` folder is accessible from the menu. It is recommended to keep **only one `.lydrc` file** in that folder at a time. If you need to switch between DRC scripts for different stacks, simply replace the file.

**Running the DRC in KLayout:**

1. Open your GDS layout in KLayout.
2. Go to **Tools → DRC**.
3. Click **Edit DRC Script** and select the `.lydrc` file corresponding to your process stack.
4. Run the script. The results will appear in a dedicated DRC results window, where violations are listed and can be highlighted in the layout.

> **Important:** Not every flagged violation necessarily needs to be corrected — some rules may be advisory or context-dependent. Conversely, the DRC script does not guarantee that all possible design errors are caught. Always review results in the context of your specific design intent and consult Luxtelligence if in doubt.

## Examples

After installing the PDK, you can verify that it is working correctly by running the Jupyter notebooks in the [docs/notebooks](https://github.com/Luxtelligence/lxt_pdk_gf/tree/main/docs/notebooks) folder.

## Documentation

- [PDK documentation](https://luxtelligence.github.io/lxt_pdk_gf/)
- [gdsfactory documentation](https://gdsfactory.github.io/gdsfactory/)
