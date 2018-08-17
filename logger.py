


class Logger:

    def __init__(self, out_dir, tags=None):
        self.lines = []

        if tags is None:
            tags = []
            
        self.tags = tags

    def write(self, msg):
        self.lines.append(msg)

    def log_target_info(self, msg):
        self.lines.append('Target info: {}'.format(msg))

    def log_data(self, data):
        self.lines.append('Logged data... TODO')
        # TODO: actually log the data

    def log_classifier_performance(self, msg, classifier_name):
        self.lines.append('Classifier: {}.\nPerformance: {}'.format(classifier_name, msg))

    def log_attack_performance(self, msg):
        self.lines.append('Attack performance: {}'.format(msg))

    def close(self):
        # TODO: this
        for tag in self.tags:
            print(tag)

        for line in self.lines:
            print(line)