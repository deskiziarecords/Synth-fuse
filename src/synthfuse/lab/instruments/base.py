"""
Base class for Lab Instruments.
"""

class LabInstrument:
    INSTRUMENT_ID = "base_instrument"
    THERMAL_COST = 0.01

    def analyze(self, *args, **kwargs):
        raise NotImplementedError
