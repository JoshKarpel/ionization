import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

PARAMETER_TO_SYMBOL = {
    "pulse_width": r"\tau",
    "fluence": r"H",
    "amplitude": r"\mathcal{E}_0",
    "phase": r"\varphi",
    "number_of_cycles": r"N_c",
    "delta_r": r"\Delta r",
    "delta_t": r"\Delta t",
}

PARAMETER_TO_UNIT_NAME = {
    "pulse_width": "asec",
    "fluence": "Jcm2",
    "amplitude": "atomic_electric_field",
    "phase": "pi",
    "number_of_cycles": "",
    "delta_r": "bohr_radius",
    "delta_t": "asec",
}


def modulation_depth(cosine, sine):
    """
    (c-s) / (c+s)

    Parameters
    ----------
    cosine
    sine

    Returns
    -------

    """
    return (cosine - sine) / (cosine + sine)
