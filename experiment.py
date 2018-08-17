
import sys

from testbed import testbed

class AttackExperiment:
    """
    Abstracts an ML attack experiment.
    """

    def __init__(self, labels):
        """
        TODO: write doc
        :param labels:
        """

        self.tb = testbed()
        tb = self.tb
        
        tb.register_experiment(self)

        self.dataset = tb.get_dataset()
        self.preprocessor_f = tb.get_preprocessor()
        self.reverse_preprocessor_f = tb.get_preprocessor_inverse()
        self.craft_f = tb.get_craft()
        self.postprocessor_f = tb.get_postprocessor()
        self.apply_f = tb.get_apply_function()
        self.surrogate_model = tb.get_surrogate_model()
        self.victim_model = tb.get_victim_model()
        self.target_picker = tb.get_target_picker()
        self.labels = labels


    # TODO: different logger for batch of experiments. responsible with statistics. Should TB do this? How should it do it?
    def run_once(self, experiment_index=0, log_instances=False):
        """
        TODO: write doc
        :return:
        """
        # Create a log file for this experiment. All subsequent logging done in the
        # custom functions (that get the logger through the testbed, i.e., tb.logger()) will write to this log file.
        #tb = testbed()

        logger = self.tb.new_experiment_log(self.labels, experiment_index)

        # The train/test/craft dataset used by the surrogate (attacker) model.
        surrogate_train_dataset = self.preprocessor_f(self.dataset.train())
        surrogate_test_dataset = self.preprocessor_f(self.dataset.test())
        surrogate_craft_dataset = self.preprocessor_f(self.dataset.craft())

        # The surrogate model - needs to be reset before training. The train method abstracts specifics
        # such as fine-tuning, pre-trained models, etc.

        self.surrogate_model.reset()
        self.surrogate_model.train(surrogate_train_dataset)

        surrogate_performance = self.surrogate_model.test(surrogate_test_dataset)

        # Log the surrogate performance.
        logger.log_classifier_performance(surrogate_performance, 'Surrogate')

        # Information about attack target: source/target classes, target instance, etc. TODO: is test dataset ok enough?
        target_info = self.target_picker(self.surrogate_model, surrogate_test_dataset)
        logger.log_target_info(target_info)

        # The base and adversarial instances. The craft function abstracts the base instance choice.
        base_instances, adversarial_instances = self.craft_f(self.surrogate_model, surrogate_craft_dataset, target_info)
        #print (type(adversarial_instances))
        #print (surrogate_test_dataset.shape)
        #exit()

        if log_instances: logger.log_data(base_instances, 'Base Instances')
        if log_instances: logger.log_data(adversarial_instances, 'Adversarial Instances')

        # Post-process the adversarial instances to emulate how the victim would receive them.
        reverse_preprocessed_adversarial_instances = self.reverse_preprocessor_f(adversarial_instances)
        post_processed_adversarial_instances = self.postprocessor_f(reverse_preprocessed_adversarial_instances, base_instances)

        if log_instances: logger.log_data(post_processed_adversarial_instances, 'Post Adversarial Instances')

        # The victim model - needs to be reset before training.
        self.victim_model.reset()
        self.victim_model.train(self.dataset.train())
        victim_performance = self.victim_model.test(self.dataset.test())

        # Log the victim model performance.
        logger.log_classifier_performance(victim_performance, 'Victim')

        # Apply the attack. The apply function abstracts the nature of the attack (e.g. poisoning vs evasion,
        # targeted vs indiscriminate, etc). The apply function is also responsible with returning attack performance.
        attack_performance = self.apply_f(self.surrogate_model, self.victim_model, self.dataset,
                                          post_processed_adversarial_instances,target_info)

        # Log the attack performance.
        logger.log_attack_performance(attack_performance)

        # TODO: adapt to parallel execution and avoid possible race condition?
        logger.close()