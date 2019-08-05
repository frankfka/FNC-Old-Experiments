

# Superclass for feature generators
class FeatureGenerator(object):

    def __init__(self, name):
        self.name = name

    # Generates features from a data object, saves to pickle if pkl_path is given
    def generate(self, data, pkl_path):
        pass

    # Load features from pkl
    def load(self, pkl_path):
        pass
