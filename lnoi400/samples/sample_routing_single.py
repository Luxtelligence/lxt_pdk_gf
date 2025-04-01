from functools import partial

import gdsfactory as gf
from lnoi400 import cells, tech



if __name__ == "__main__":
    c = gf.Component()

    mzm =c << cells.mzm_unbalanced()
    ec1 = c << cells.double_linear_inverse_taper()

    ec2 = c << cells.double_linear_inverse_taper()
    ec2.mirror()
    ec2.xmin = mzm.xmax + 900
    ec2.ymin = mzm.ymax + 900

    gf.routing.route_single(c, mzm['o2'], ec2['o2'], straight="straight_rwg1000", cross_section="xs_rwg1000", bend='L_turn_bend')
    c.pprint_ports()
    c.show()
