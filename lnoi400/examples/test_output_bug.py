import gdsfactory as gf

mmi = gf.get_component("mmi1x2_optimized1550")
mmi.write("mmi_test.gds")
