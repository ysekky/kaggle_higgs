__author__ = 'ysekky'


from learner_base import LearnerBase
from sklearn.svm import SVC

class SimpleRandomForest(LearnerBase):


    def learning(self, training_data, feature_names, **kwargs):
        x = self.create_feature(training_data, feature_names)
        y = self.encode_training_class(training_data)
        self.learner = SVC(kernel="rbf", C=0.025, probability=True)
        self.learner.fit(x, y)

    def predict(self, test_data, feature_names):
        x = self.create_feature(test_data, feature_names)
        result = []
        probabilities = []
        for y, prob in self.learner.predict_proba(x):
            result.append(y)
            probabilities.append(prob)
        rank_order = [p_index for p_index, p in sorted(enumerate(probabilities), lambda d:-d[1])]
        self.submit(test_data, result, rank_order)