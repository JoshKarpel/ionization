class IonizationException(Exception):
    pass


class InvalidPotentialParameter(IonizationException):
    pass


class InvalidMaskParameter(IonizationException):
    pass


class IllegalQuantumState(IonizationException):
    """An exception indicating that there was an attempt to cosntruct a state with an illegal quantum number."""
    pass

class InvalidChoice(IonizationException):
    pass
