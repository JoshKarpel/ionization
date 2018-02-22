class IonizationException(Exception):
    pass


class InvalidPotentialParameter(IonizationException):
    pass


class InvalidMaskParameter(IonizationException):
    pass


class IllegalQuantumState(IonizationException):
    """An exception indicating that there was an attempt to construct a state with an illegal quantum number."""
    pass


class InvalidChoice(IonizationException):
    pass


class InvalidWrappingDirection(IonizationException):
    pass


class MissingDatastore(IonizationException):
    pass


class UnknownData(IonizationException):
    pass


class DuplicateDatastores(IonizationException):
    pass
