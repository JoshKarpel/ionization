import simulacra as si
import ionization as ion

TUNNELING_MODEL_TYPES = [
    ion.tunneling.LandauRate,
    ion.tunneling.KeldyshRate,
    ion.tunneling.MulserRate,
    ion.tunneling.PosthumusRate,
    ion.tunneling.ADKRate,
    # ion.tunneling.ADKExtendedToBSIRate,
]
