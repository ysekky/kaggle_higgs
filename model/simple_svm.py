__author__ = 'ysekky'


from learner_base import LearnerBase
from sklearn.svm import SVC

class SimpleSVM(LearnerBase):


    def learning(self, training_data, feature_names, **kwargs):
        x = self.create_feature(training_data, feature_names)
        y = self.encode_training_class(training_data)
        self.learner = SVC(kernel="rbf", C=0.025)
        self.learner.fit(x, y)

    def predict(self, test_data, feature_names):
        x = self.create_feature(test_data, feature_names)
        result = [y for y in self.learner.predict(x)]
        rank_order = range(1, len(result)+1)
        self.submit(test_data, result, rank_order)