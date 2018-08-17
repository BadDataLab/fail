from experiment import AttackExperiment
from utils import targeted_evasion_attack_apply, SourceTargetClassPicker # TODO: cleverhans_craft_wrapper
from attacks import jsma_craft

class JSMAUnconstrainedExperiment(AttackExperiment):

    def __init__(self, labels):
        super().__init__(labels)
        self.craft_f = jsma_craft
        self.apply_f = targeted_evasion_attack_apply
        self.target_picker = SourceTargetClassPicker(0, 5)


