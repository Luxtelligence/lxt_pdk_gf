import sax
import sax.models

sax.set_port_naming_strategy("optical")


def straight(
    wl=1.55, wl0=1.55, neff=2.34, ng=3.4, length=10.0, loss_dB_cm=1.0
) -> sax.SType:
    """Dispersive straight waveguide model.

    ```{svgbob}
    in0             out0
     o1 =========== o2
    ```

    Args:
        wl: The wavelength in micrometers.
        wl0: The center wavelength used for dispersion calculation.
        neff: The Effective refractive index at the center wavelength.
        ng: The Group refractive index at the center wavelength.
        length: The length of the waveguide in micrometers.
        loss_dB_cm: The Propagation loss in dB/cm.

    Returns:
        S-matrix dictionary containing the complex transmission coefficient.

    Examples:
        Lossless waveguide:

        ```python
        # mkdocs: render
        import matplotlib.pyplot as plt
        import numpy as np
        import sax

        sax.set_port_naming_strategy("optical")

        wl = sax.wl_c()
        s = sax.models.straight(wl=wl, coupling=0.3)
        thru = np.abs(s[("o1", "o2")]) ** 2
        plt.figure()
        plt.plot(wl, thru, label="thru")
        plt.xlabel("Wavelength [μm]")
        plt.ylabel("Power")
        plt.legend()
        ```
    """
    return sax.models.straight(
        wl=wl, wl0=wl0, neff=neff, ng=ng, length=length, loss_dB_cm=loss_dB_cm
    )


def bend_euler(
    wl=1.55, wl0=1.55, neff=2.34, ng=3.4, length=10.0, loss_dB_cm=1.0
) -> sax.SType:
    """Dispersive straight waveguide model.

    ```{svgbob}
    in0             out0
     o1 =========== o2
    ```

    Args:
        wl: The wavelength in micrometers.
        wl0: The center wavelength used for dispersion calculation.
        neff: The Effective refractive index at the center wavelength.
        ng: The Group refractive index at the center wavelength.
        length: The length of the waveguide in micrometers.
        loss_dB_cm: The Propagation loss in dB/cm.

    Returns:
        S-matrix dictionary containing the complex transmission coefficient.

    Examples:
        Lossless waveguide:

        ```python
        # mkdocs: render
        import matplotlib.pyplot as plt
        import numpy as np
        import sax

        sax.set_port_naming_strategy("optical")

        wl = sax.wl_c()
        s = sax.models.straight(wl=wl, coupling=0.3)
        thru = np.abs(s[("o1", "o2")]) ** 2
        plt.figure()
        plt.plot(wl, thru, label="thru")
        plt.xlabel("Wavelength [μm]")
        plt.ylabel("Power")
        plt.legend()
        ```
    """
    return sax.models.straight(
        wl=wl, wl0=wl0, neff=neff, ng=ng, length=length, loss_dB_cm=loss_dB_cm
    )


def straight_rwg1000(
    wl=1.55, wl0=1.55, neff=2.34, ng=3.4, length=10.0, loss_dB_cm=1.0
) -> sax.SType:
    """Dispersive straight waveguide model.

    ```{svgbob}
    in0             out0
     o1 =========== o2
    ```

    Args:
        wl: The wavelength in micrometers.
        wl0: The center wavelength used for dispersion calculation.
        neff: The Effective refractive index at the center wavelength.
        ng: The Group refractive index at the center wavelength.
        length: The length of the waveguide in micrometers.
        loss_dB_cm: The Propagation loss in dB/cm.

    Returns:
        S-matrix dictionary containing the complex transmission coefficient.

    Examples:
        Lossless waveguide:

        ```python
        # mkdocs: render
        import matplotlib.pyplot as plt
        import numpy as np
        import sax

        sax.set_port_naming_strategy("optical")

        wl = sax.wl_c()
        s = sax.models.straight(wl=wl, coupling=0.3)
        thru = np.abs(s[("o1", "o2")]) ** 2
        plt.figure()
        plt.plot(wl, thru, label="thru")
        plt.xlabel("Wavelength [μm]")
        plt.ylabel("Power")
        plt.legend()
        ```
    """
    return sax.models.straight(
        wl=wl, wl0=wl0, neff=neff, ng=ng, length=length, loss_dB_cm=loss_dB_cm
    )
