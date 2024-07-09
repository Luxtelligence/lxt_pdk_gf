

Luxtelligence provides a library of components that have been fabricated in the reference material stack, and whose performance has been tested and validated. Here follows a list of the available parametric cells (gdsfactory.Component objects):


Cells
=============================


CPW_pad_linear
----------------------------------------------------

.. autofunction:: lnoi400.cells.CPW_pad_linear

.. plot::
  :include-source:

  import lnoi400

  c = lnoi400.cells.CPW_pad_linear(start_width=80.0, length_straight=10.0, length_tapered=190.0, cross_section='xs_uni_cpw')
  c.plot()



L_turn_bend
----------------------------------------------------

.. autofunction:: lnoi400.cells.L_turn_bend

.. plot::
  :include-source:

  import lnoi400

  c = lnoi400.cells.L_turn_bend(radius=80.0, p=1.0, with_arc_floorplan=True, cross_section='xs_rwg1000')
  c.plot()



S_bend_vert
----------------------------------------------------

.. autofunction:: lnoi400.cells.S_bend_vert

.. plot::
  :include-source:

  import lnoi400

  c = lnoi400.cells.S_bend_vert(v_offset=25.0, h_extent=100.0, dx_straight=5.0, cross_section='xs_rwg1000')
  c.plot()



U_bend_racetrack
----------------------------------------------------

.. autofunction:: lnoi400.cells.U_bend_racetrack

.. plot::
  :include-source:

  import lnoi400

  c = lnoi400.cells.U_bend_racetrack(v_offset=90.0, p=1.0, with_arc_floorplan=True, cross_section='xs_rwg3000')
  c.plot()



bend_S_spline
----------------------------------------------------

.. autofunction:: lnoi400.cells.bend_S_spline

.. plot::
  :include-source:

  import lnoi400

  c = lnoi400.cells.bend_S_spline(size=(100.0, 30.0), cross_section='xs_rwg1000', npoints=201)
  c.plot()



chip_frame
----------------------------------------------------

.. autofunction:: lnoi400.cells.chip_frame

.. plot::
  :include-source:

  import lnoi400

  c = lnoi400.cells.chip_frame(size=(10000, 5000), exclusion_zone_width=50)
  c.plot()



double_linear_inverse_taper
----------------------------------------------------

.. autofunction:: lnoi400.cells.double_linear_inverse_taper

.. plot::
  :include-source:

  import lnoi400

  c = lnoi400.cells.double_linear_inverse_taper(cross_section_start='xs_swg250', cross_section_end='xs_rwg1000', lower_taper_length=120.0, lower_taper_end_width=2.05, upper_taper_start_width=0.25, upper_taper_length=240.0, slab_removal_width=20.0, input_ext=0.0)
  c.plot()



eo_phase_shifter
----------------------------------------------------

.. autofunction:: lnoi400.cells.eo_phase_shifter

.. plot::
  :include-source:

  import lnoi400

  c = lnoi400.cells.eo_phase_shifter(rib_core_width_modulator=2.5, taper_length=100.0, modulation_length=7500.0, rf_central_conductor_width=10.0, rf_ground_planes_width=180.0, rf_gap=4.0, draw_cpw=True)
  c.plot()



mmi1x2_optimized1550
----------------------------------------------------

.. autofunction:: lnoi400.cells.mmi1x2_optimized1550

.. plot::
  :include-source:

  import lnoi400

  c = lnoi400.cells.mmi1x2_optimized1550(width_mmi=6.0, length_mmi=26.75, width_taper=1.5, length_taper=25.0, port_ratio=0.55, cross_section='xs_rwg1000')
  c.plot()



mmi2x2optimized1550
----------------------------------------------------

.. autofunction:: lnoi400.cells.mmi2x2optimized1550

.. plot::
  :include-source:

  import lnoi400

  c = lnoi400.cells.mmi2x2optimized1550(width_mmi=5.0, length_mmi=76.5, width_taper=1.5, length_taper=25.0, port_ratio=0.7, cross_section='xs_rwg1000')
  c.plot()



mzm_unbalanced
----------------------------------------------------

.. autofunction:: lnoi400.cells.mzm_unbalanced

.. plot::
  :include-source:

  import lnoi400

  c = lnoi400.cells.mzm_unbalanced(modulation_length=7500.0, lbend_tune_arm_reff=75.0, rf_pad_start_width=80.0, rf_central_conductor_width=10.0, rf_ground_planes_width=180.0, rf_gap=4.0, rf_pad_length_straight=10.0, rf_pad_length_tapered=190.0)
  c.plot()



straight_rwg1000
----------------------------------------------------

.. autofunction:: lnoi400.cells.straight_rwg1000

.. plot::
  :include-source:

  import lnoi400

  c = lnoi400.cells.straight_rwg1000(length=10.0)
  c.plot()



straight_rwg3000
----------------------------------------------------

.. autofunction:: lnoi400.cells.straight_rwg3000

.. plot::
  :include-source:

  import lnoi400

  c = lnoi400.cells.straight_rwg3000(length=10.0)
  c.plot()



uni_cpw_straight
----------------------------------------------------

.. autofunction:: lnoi400.cells.uni_cpw_straight

.. plot::
  :include-source:

  import lnoi400

  c = lnoi400.cells.uni_cpw_straight(length=3000.0, cross_section='xs_uni_cpw', bondpad='CPW_pad_linear')
  c.plot()
