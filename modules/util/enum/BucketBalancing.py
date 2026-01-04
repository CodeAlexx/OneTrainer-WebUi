from enum import Enum


class BucketBalancing(Enum):
    OFF = 'OFF'
    OVERSAMPLE = 'OVERSAMPLE'
    WEIGHTED = 'WEIGHTED'

    def __str__(self):
        return self.value
