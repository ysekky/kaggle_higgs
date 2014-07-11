__author__ = 'ysekky'


from learner_base import LearnerBase
from sklearn.svm import SVC

class SimpleSVM(LearnerBase):


    def __train(self, x, y, **kwargs):
        """
        """
        self.learner = SVC(kernel="rbf", C=0.025)
        self.learner.fit(x, y)

    def __predict(self, x):
        result = []
        probabilities = []
        for y, prob in self.learner.predict_proba(x):
            result.append(y)
            probabilities.append(prob)
        rank_order = [p_index for p_index, p in sorted(enumerate(probabilities), lambda d:-d[1])]
        return result, rank_order