# this class must have simple fields in order to be serialized
import random

from DeepJanus.core.config import Config

class SEDEConfig(Config):
    def __init__(self, ub, lb, centroid, caseFile):
        super().__init__()
        self.upper_bound = ub
        self.lower_bound = lb
        self.num_control_nodes = len(ub)
        self.centroid = centroid
        self.caseFile = caseFile
        self.globalCounter = random.randint(1, 999999999)
