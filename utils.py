import numpy as np
from testbed import testbed

"""
self.apply_f(self.surrogate_model, self.victim_model, self.dataset,
                                          post_processed_adversarial_instances)
"""


class SourceTargetClassPicker:
    def __init__(self, source, target):
        self.source = source
        self.target = target
        self.counter = 0

    def __call__(self, surrogate_model, surrogate_test_dataset):
        src_instances = surrogate_test_dataset.y == self.source
        target_instance = src_instances[self.counter]
        self.counter += 1

        if self.counter >= len(src_instances):
            self.counter = 0

            tb = testbed()
            tb.logger().write('Target instance counter was reset to 0.')

        return {
            'target_instance': target_instance,
            'source_class': self.source,
            'target_class': self.target
        }


def targeted_evasion_attack_apply(surrogate_model, victim_model, dataset, adversarial_instances, target_info):
    """
    Applies the adversarial instances at test time. TODO: fill in doc
    :param surrogate:
    :param victim:
    :param dataset:
    :param adversarial_instances:
    :return:
    """
    target_class = target_info['target_class']

    surrogate_acc = surrogate_model.test(adversarial_instances)
    victim_acc = victim_model.test(adversarial_instances)

    return {'surrogate_acc': surrogate_acc,'victim_acc': victim_acc}
    # TODO get actual predictions 

    surrogate_ys = adversarial_instances.y.tolist()
    victim_ys = adversarial_instances.y.tolist()

    successfull_xs = adversarial_instances[np.argwhere(surrogate_ys == np.repeat(target_class,
                                                                                 repeats=len(surrogate_ys)))]

    victim_actual_ys = victim_model.test(successfull_xs)

    return {
        'perceived_success_rate': np.sum(surrogate_ys == target_class) / len(surrogate_ys),
        'potential_success_rate': np.sum(victim_ys == target_class) / len(victim_ys),
        'actual_success_rate': np.sum(victim_actual_ys == target_class) / len(victim_actual_ys)
    }


def targeted_poisoning_attack_apply(surrogate_model, victim_model, dataset, adversarial_instances, target_info):
    """
    Applies the adversarial instances at test time. TODO: fill in doc
    :param surrogate:
    :param victim:
    :param dataset:
    :param adversarial_instances:
    :return:
    """
    target_class = target_info['target_class']

    surrogate_acc = surrogate_model.test(adversarial_instances)
    victim_acc = victim_model.test(adversarial_instances)

    surrogate_ys = adversarial_instances.y.tolist()
    victim_ys = adversarial_instances.y.tolist()

    success_idx = np.argwhere(surrogate_ys == np.repeat(target_class, repeats=len(surrogate_ys)))
    success_idx = np.squeeze(success_idx)
    
    successfull_xs = adversarial_instances.get(success_idx)

    victim_actual_ys = victim_model.test(successfull_xs)

    return {
        'perceived_success_rate': np.sum(surrogate_ys == target_class) / len(surrogate_ys),
        'potential_success_rate': np.sum(victim_ys == target_class) / len(victim_ys),
        'actual_success_rate': np.sum(victim_actual_ys == target_class) / len(victim_actual_ys)
    }
