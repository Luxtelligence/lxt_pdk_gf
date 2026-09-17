import gdsfactory as gf

from _utils.spline import (
    bend_S_spline_varying_width,
)
from ltoi300.tech import xs_rwg900

# ---------------------------------------------------------------------------
# Default parameter values — defined once, shared by all sub-cells and the
# top-level assembler.  Each sub-cell function declares only the parameters
# that are relevant to it, all with these defaults.
# ---------------------------------------------------------------------------
_DEFAULTS = dict(
    Lstr=10.0,  # Unified straight length (replaces Lentr / Lmid half / Lcross_ext) (µm)
    w1=1.0,  # Ridge width after taper 1 (µm)
    Ltp1=100.0,  # Length of taper 1: 0.9 µm → w1 (µm)
    w2=1.7,  # Ridge width after taper 2 (µm)
    Ltp2=500.0,  # Length of taper 2: w1 → w2 (µm)
    w3=2.0,  # Ridge width at mid-straight / coupler (µm)
    Ltp3=100.0,  # Length of taper 3: w2 → w3 (µm)
    Lcoupler=26.0,  # Length of coupler straight at w3 (µm)
    Ltp4=40.0,  # Length of exit taper: w3 → 0.9 µm (µm)
    w4=0.6,  # Cross-port start width — narrow (left) end (µm)
    w5=0.85,  # Cross-port end width   — wide  (right) end (µm)
    coupler_gap=0.45,  # Edge-to-edge gap between bus and cross-port at w5 (µm)
)

D = _DEFAULTS  # short alias used in signatures below


# ---------------------------------------------------------------------------
# 1.  PSR Evolution Taper
#     Entrance straight  →  Taper 1  →  Taper 2  →  Taper 3  →  half-mid straight
#
#     o1 : input  (west, xs_rwg900 width 0.9 µm)
#     o2 : output (east, xs_rwg900 width w3)
# ---------------------------------------------------------------------------
@gf.cell
def psr_evolution_taper(
    Lstr: float = D["Lstr"],  # Entrance straight + half-mid straight length each (µm)
    w1: float = D["w1"],
    Ltp1: float = D["Ltp1"],
    w2: float = D["w2"],
    Ltp2: float = D["Ltp2"],
    w3: float = D["w3"],
    Ltp3: float = D["Ltp3"],
) -> gf.Component:
    """
    Mode-evolution taper for the PSR bus waveguide.

    Sections (left → right):
      1. Entrance straight : xs_rwg900 (0.9 µm),  length = Lstr
      2. Taper 1           : 0.9 µm  → w1,         length = Ltp1
      3. Taper 2           : w1      → w2,          length = Ltp2
      4. Taper 3           : w2      → w3,          length = Ltp3
      5. Half-mid straight : w3,                    length = Lstr

    The half-mid straight at the output gives a well-defined reference
    cross-section (w3) for connecting to the coupler section.

    Ports:
        o1 — input  (west, 0.9 µm)
        o2 — output (east, w3)
    """
    c = gf.Component()

    # 1. Entrance straight
    s_entr = c << gf.components.straight(length=Lstr, cross_section=xs_rwg900())

    # 2. Taper 1: 0.9 µm → w1
    tp1 = c << gf.components.taper_cross_section(
        cross_section1=xs_rwg900(),
        cross_section2=xs_rwg900(width=w1),
        length=Ltp1,
    )
    tp1.connect("o1", s_entr.ports["o2"])

    # 3. Taper 2: w1 → w2
    tp2 = c << gf.components.taper_cross_section(
        cross_section1=xs_rwg900(width=w1),
        cross_section2=xs_rwg900(width=w2),
        length=Ltp2,
    )
    tp2.connect("o1", tp1.ports["o2"])

    # 4. Taper 3: w2 → w3
    tp3 = c << gf.components.taper_cross_section(
        cross_section1=xs_rwg900(width=w2),
        cross_section2=xs_rwg900(width=w3),
        length=Ltp3,
    )
    tp3.connect("o1", tp2.ports["o2"])

    # 5. Half-mid straight at w3
    s_half_mid = c << gf.components.straight(
        length=Lstr, cross_section=xs_rwg900(width=w3)
    )
    s_half_mid.connect("o1", tp3.ports["o2"])

    c.add_port(name="o1", port=s_entr.ports["o1"])
    c.add_port(name="o2", port=s_half_mid.ports["o2"])
    return c


# ---------------------------------------------------------------------------
# 2.  PSR Coupler
#     Half-mid straight  →  coupler straight (bus + cross-port taper)
#
#     o1 : bus input   (west, w3)
#     o2 : bus output  (east, w3)
#     o3 : cross-port output (east end of cross-port taper, width w5)
# ---------------------------------------------------------------------------
@gf.cell
def psr_coupler(
    Lstr: float = D["Lstr"],
    w3: float = D["w3"],
    Lcoupler: float = D["Lcoupler"],
    w4: float = D["w4"],
    w5: float = D["w5"],
    coupler_gap: float = D["coupler_gap"],
) -> gf.Component:
    """
    Directional-coupler section of the PSR.

    Sections (left → right):
      1. Half-mid straight : w3,              length = Lstr
      2. Coupler straight  : w3 (bus),        length = Lcoupler
         Cross-port taper  : w4 → w5,         length = Lcoupler
         (cross-port is offset +y by w3/2 + coupler_gap + w5/2)

    Ports:
        o1 — bus input          (west, w3)
        o2 — bus output         (east, w3)   [directly at coupler section end]
        o3 — cross-port output  (east, w5)   [directly at cross taper end]
    """
    c = gf.Component()

    # 1. Half-mid straight
    s_half_mid = c << gf.components.straight(
        length=Lstr, cross_section=xs_rwg900(width=w3)
    )

    # 2. Bus coupler straight
    s_coup = c << gf.components.straight(
        length=Lcoupler, cross_section=xs_rwg900(width=w3)
    )
    s_coup.connect("o1", s_half_mid.ports["o2"])

    # 3. Cross-port taper (w4 → w5), x-aligned with s_coup left edge, offset +y
    y_cross = w3 / 2 + coupler_gap + w5 / 2
    s_cross = c << gf.components.taper_cross_section(
        cross_section1=xs_rwg900(width=w4),
        cross_section2=xs_rwg900(width=w5),
        length=Lcoupler,
    )
    s_cross.dmove(
        (
            s_coup.ports["o1"].dx - s_cross.ports["o1"].dx,
            s_coup.dcenter[1] + y_cross - s_cross.dcenter[1],
        )
    )

    c.add_port(name="o1", port=s_half_mid.ports["o1"])
    c.add_port(name="o2", port=s_coup.ports["o2"])
    c.add_port(name="o3", port=s_cross.ports["o2"])
    return c


# ---------------------------------------------------------------------------
# 3.  PSR Mode Router
#     Splits the converted TM→TE mode (cross-port) away from the bus.
#
#     Receives the bus (width w3) and the cross-port (width w5) from the
#     coupler; routes them to separate output ports at the same x-coordinate.
#
#     Bus path    : straight Lstr  →  exit taper Ltp4  →  straight (auto)
#     Cross path  :                   S-bend            →  straight Lstr
#
#     o1 : bus input        (west, w3)
#     o2 : cross-port input (west, w5)   [must be placed by caller]
#     o3 : cross-port output (east, 0.9 µm)
#     o4 : bus output        (east, 0.9 µm)
# ---------------------------------------------------------------------------
@gf.cell
def psr_mode_router(
    Lstr: float = D["Lstr"],
    w3: float = D["w3"],
    w5: float = D["w5"],
    Ltp4: float = D["Ltp4"],
    coupler_gap: float = D["coupler_gap"],
    size: tuple[float, float] = (
        58,
        14.5,
    ),  # modify only for simulations to reduce costs
) -> gf.Component:
    """
    Mode-router section of the PSR.

    Separates the converted TE1 cross-port signal from the bus waveguide.
    Both the exit taper (bus) and S-bend (cross-port) start at the left
    edge of this cell.  The S-bend input is positioned at the same
    y-offset as the coupler cross-port exit:

        y_cross = w3/2 + coupler_gap + w5/2   (centre-to-centre, from bus)

    Bus path (left → right):
      1. Exit taper       : w3 → 0.9 µm,    length = Ltp4
      2. Through straight : length = auto    (aligns o4 flush with o3)

    Cross-port path (left → right):
      1. Exit S-bend      : w5 → 0.9 µm,    curves away from bus
      2. Exit straight    : 0.9 µm,          length = Lstr

    Ports:
        o1 — bus input         (west, w3)     — connect to coupler o2
        o2 — cross-port input  (west, w5)     — connect to coupler o3
        o3 — cross-port output (east, 0.9 µm)
        o4 — bus output        (east, 0.9 µm)
    """
    c = gf.Component()

    # --- Bus path ---
    # Exit taper: w3 → 0.9 µm (starts at left edge of cell, x=0, y=0)
    tp4 = c << gf.components.taper_cross_section(
        cross_section1=xs_rwg900(width=w3),
        cross_section2=xs_rwg900(),
        length=Ltp4,
    )

    # --- Cross-port path ---
    # S-bend: w5 → 0.9 µm, positioned at coupler_gap above bus
    y_cross = w3 / 2 + coupler_gap + w5 / 2
    s_sbend = c << bend_S_spline_varying_width(
        size=size,
        cross_section1=xs_rwg900(width=w5),
        cross_section2=xs_rwg900(),
        layer_ridge="LT_RIDGE",
        layer_slab="LT_SLAB",
    )
    # Align S-bend input (o1) to the same x as tp4 input, offset +y by y_cross
    s_sbend.dmove(
        (
            tp4.ports["o1"].dx - s_sbend.ports["o1"].dx,
            tp4.ports["o1"].dy + y_cross - s_sbend.ports["o1"].dy,
        )
    )

    # Post-S-bend exit straight at 0.9 µm
    s_cross_ext = c << gf.components.straight(length=Lstr, cross_section=xs_rwg900())
    s_cross_ext.connect("o1", s_sbend.ports["o2"])

    # Auto-length through-port straight so o4 aligns with o3
    x_target = s_cross_ext.ports["o2"].dx
    Lthrough = x_target - tp4.ports["o2"].dx
    if Lthrough < 0:
        raise ValueError(
            f"psr_mode_router: computed through-straight length {Lthrough:.3f} µm is "
            "negative. Increase Ltp4 or Lstr so the bus taper does not overshoot the "
            "cross-port exit."
        )
    s_through = c << gf.components.straight(length=Lthrough, cross_section=xs_rwg900())
    s_through.connect("o1", tp4.ports["o2"])

    c.add_port(name="o1", port=tp4.ports["o1"])
    c.add_port(name="o2", port=s_sbend.ports["o1"])  # cross-port input
    c.add_port(name="o3", port=s_cross_ext.ports["o2"])
    c.add_port(name="o4", port=s_through.ports["o2"])
    return c


# ---------------------------------------------------------------------------
# 4.  Coupler + Router  (simulation sub-assembly)
#     Connects psr_coupler → psr_mode_router without the evolution taper.
#     Useful for isolated FDTD / EME simulation of the coupling + routing stage.
#
#     Ports:
#         o1 — bus input         (west, w3)      ← connect to evolution taper o2
#         o2 — cross-port output (east, 0.9 µm)  ← converted TM→TE
#         o3 — through output    (east, 0.9 µm)  ← residual TE
# ---------------------------------------------------------------------------
@gf.cell
def psr_coupler_router(
    Lstr: float = D["Lstr"],
    w3: float = D["w3"],
    Lcoupler: float = D["Lcoupler"],
    w4: float = D["w4"],
    w5: float = D["w5"],
    coupler_gap: float = D["coupler_gap"],
    Ltp4: float = D["Ltp4"],
    size: tuple[float, float] = (58, 14.5),
) -> gf.Component:
    """
    Simulation sub-assembly: psr_coupler + psr_mode_router.

    Assembles the directional-coupler and mode-routing sections without the
    evolution taper, so the coupling + routing region can be simulated in
    isolation.

    Sections (left → right):
      ┌──────────────┐   ┌──────────────────┐
      │  psr_coupler  │   │  psr_mode_router │
      │ o1 ─────► o2 ├───┤ o1 ─────────► o4 │ → o3 (through, 0.9 µm)
      │           o3 ├───┤ o2 ─────────► o3 │ → o2 (cross,   0.9 µm)
      └──────────────┘   └──────────────────┘

    Ports:
        o1 — bus input         (west, w3)
        o2 — cross-port output (east, 0.9 µm)
        o3 — through output    (east, 0.9 µm)
    """
    c = gf.Component()

    coup = c << psr_coupler(
        Lstr=Lstr,
        w3=w3,
        Lcoupler=Lcoupler,
        w4=w4,
        w5=w5,
        coupler_gap=coupler_gap,
    )

    router = c << psr_mode_router(
        Lstr=Lstr,
        w3=w3,
        w5=w5,
        Ltp4=Ltp4,
        coupler_gap=coupler_gap,
        size=size,
    )
    router.connect("o1", coup.ports["o2"])
    router.connect("o2", coup.ports["o3"])

    c.add_port(name="o1", port=coup.ports["o1"])  # bus input
    c.add_port(name="o2", port=router.ports["o3"])  # cross output
    c.add_port(name="o3", port=router.ports["o4"])  # through output
    return c


# ---------------------------------------------------------------------------
# 5.  Polarisation Splitter-Rotator (top-level assembler)
#     Connects the three sub-cells in sequence.
#
#     Ports:
#         o1 — input  (west, 0.9 µm)
#         o2 — cross-port output (east, 0.9 µm)   ← converted TM→TE
#         o3 — through-port output (east, 0.9 µm) ← residual TE
# ---------------------------------------------------------------------------
@gf.cell
def polarisation_splitter_rotator(
    # Shared straight length (entrance / half-mid / pre-taper / cross-exit)
    Lstr: float = D["Lstr"],
    # Evolution taper parameters
    w1: float = D["w1"],
    Ltp1: float = D["Ltp1"],
    w2: float = D["w2"],
    Ltp2: float = D["Ltp2"],
    w3: float = D["w3"],
    Ltp3: float = D["Ltp3"],
    # Coupler parameters
    Lcoupler: float = D["Lcoupler"],
    w4: float = D["w4"],
    w5: float = D["w5"],
    coupler_gap: float = D["coupler_gap"],
    # Mode-router parameters
    Ltp4: float = D["Ltp4"],
) -> gf.Component:
    """
    Polarisation Splitter-Rotator (PSR) — full device.

    Assembles three sub-cells in sequence:
      ┌────────────────────┐   ┌──────────────┐   ┌──────────────────┐
      │  psr_evolution     │   │  psr_coupler  │   │  psr_mode_router │
      │  _taper            ├───┤               ├───┤                  │
      │  o1 ──────────► o2 │   │ o1 ─────► o2  │   │ o1 ─────────► o4 │ → o3 (through)
      └────────────────────┘   │           o3  │   │ o2 ─────────► o3 │ → o2 (cross)
                                └──────────────┘   └──────────────────┘

    Parameters:
        Lstr:        Unified straight-section length used as:
                       - entrance straight in psr_evolution_taper
                       - half-mid straight in psr_evolution_taper & psr_coupler
                       - coupler bus/cross-port exit straights in psr_coupler
                       - cross-port exit straight in psr_mode_router
        w1 … Ltp3:  Evolution taper geometry.
        Lcoupler, w4, w5, coupler_gap: Coupler geometry.
        Ltp4:        Exit taper length in mode router.

    Ports:
        o1 — input        (west, xs_rwg900 0.9 µm)
        o2 — cross output (east, xs_rwg900 0.9 µm) — converted TM→TE mode
        o3 — thru  output (east, xs_rwg900 0.9 µm) — residual TE mode
    """
    c = gf.Component()

    # ── Sub-cell instantiation ─────────────────────────────────────────────
    evo = c << psr_evolution_taper(
        Lstr=Lstr,
        w1=w1,
        Ltp1=Ltp1,
        w2=w2,
        Ltp2=Ltp2,
        w3=w3,
        Ltp3=Ltp3,
    )

    coup = c << psr_coupler(
        Lstr=Lstr,
        w3=w3,
        Lcoupler=Lcoupler,
        w4=w4,
        w5=w5,
        coupler_gap=coupler_gap,
    )
    coup.connect("o1", evo.ports["o2"])

    router = c << psr_mode_router(
        Lstr=Lstr, w3=w3, w5=w5, Ltp4=Ltp4, coupler_gap=coupler_gap
    )
    router.connect("o1", coup.ports["o2"])

    # Place cross-port input (o2) of router at coupler's cross-port output (o3)
    router.connect("o2", coup.ports["o3"])

    # ── Expose top-level ports ─────────────────────────────────────────────
    c.add_port(name="o1", port=evo.ports["o1"])
    c.add_port(name="o2", port=router.ports["o3"])  # cross output
    c.add_port(name="o3", port=router.ports["o4"])  # through output
    return c


# ---------------------------------------------------------------------------
# Backward-compatible alias so existing code that imports prs_waveguide
# continues to work unchanged.
# ---------------------------------------------------------------------------
@gf.cell
def prs_waveguide(
    Lentr: float = D["Lstr"],  # mapped → Lstr
    w1: float = D["w1"],
    Ltp1: float = D["Ltp1"],
    w2: float = D["w2"],
    Ltp2: float = D["Ltp2"],
    w3: float = D["w3"],
    Ltp3: float = D["Ltp3"],
    Lmid: float = D["Lstr"] * 2,  # Lmid = 2 × Lstr (half goes to evo, half to coupler)
    Lcoupler: float = D["Lcoupler"],
    Ltp4: float = D["Ltp4"],
    Lcross_ext: float = D["Lstr"],  # mapped → Lstr
    w4: float = D["w4"],
    w5: float = D["w5"],
    coupler_gap: float = D["coupler_gap"],
) -> gf.Component:
    """Backward-compatible wrapper around polarisation_splitter_rotator."""
    return polarisation_splitter_rotator(
        Lstr=Lentr,  # Lentr used as the unified Lstr
        w1=w1,
        Ltp1=Ltp1,
        w2=w2,
        Ltp2=Ltp2,
        w3=w3,
        Ltp3=Ltp3,
        Lcoupler=Lcoupler,
        w4=w4,
        w5=w5,
        coupler_gap=coupler_gap,
        Ltp4=Ltp4,
    )


if __name__ == "__main__":
    pass
    # Show each sub-cell individually, then the full device
    # psr_evolution_taper().show()
    # psr_coupler().show()
    # psr_coupler_router().show()
    # psr_mode_router().show()
    c = polarisation_splitter_rotator()
    c.show()
    # print(f"Ports: {list(c.ports.keys())}")
