from experiment import AttackExperiment
from utils import targeted_poisoning_attack_apply, SourceTargetClassPicker
from attacks import stingray_craft

class StingRayUnconstrainedExperiment(AttackExperiment):

    def __init__(self, labels):
        super().__init__(labels)
        self.craft_f = stingray_craft
        self.apply_f = targeted_poisoning_attack_apply
        self.target_picker = SourceTargetClassPicker(0, 1)


