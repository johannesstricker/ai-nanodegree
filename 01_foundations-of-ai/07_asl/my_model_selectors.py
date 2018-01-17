import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        with np.errstate(divide='ignore'):
            models = []
            for n in range(self.min_n_components, self.max_n_components + 1):
                model = self.base_model(n)
                # If there has been an error, we don't consider n number of components.
                if model is None:
                    continue
                # hmmlean may not be able to score the model, so we need to put
                # a try..catch block around it.
                try:
                    logL = model.score(self.X, self.lengths)
                    # (free transition probability count) + (free starting probability count)
                    # + (number of means) + (number of covariances)
                    # <=>
                    # (n*(n-1)) + (n-1) + (n*f) + (n*f)
                    # <=>
                    f = len(self.X[0]) if len(self.X) > 0 else 0
                    p = n**2 + 2*n*f - 1
                    BIC = (-2) * logL + p * math.log(len(self.X))
                    models.append( (model, BIC) )
                except:
                    pass
            if len(models) == 0:
                return self.base_model(self.min_n_components)
            bestModel = min(models, key = lambda x: x[1])[0]
        return bestModel


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        with np.errstate(divide='ignore'):
            models = []
            for n in range(self.min_n_components, self.max_n_components + 1):
                # Train the model on n components.
                model = self.base_model(n)
                # If there has been an error, we don't consider n number of components.
                if model is None:
                    continue
                # hmmlean may not be able to score the model, so we need to put
                # a try..catch block around it.
                try:
                    logL = model.score(self.X, self.lengths)
                    # Calculate the score of the anti-evidence as the average
                    # over scoring all other words with this model.
                    antiEvidenceScore = 0
                    for word in self.words.keys():
                        if word != self.this_word:
                            X, lengths = self.hwords[word]
                            antiEvidenceScore += model.score(X, lengths)
                    antiEvidenceScore /= (len(self.words.keys()) - 1)
                    # Calculate the final score.
                    score = logL - antiEvidenceScore
                    models.append( (model, score) )
                except:
                    pass
            if len(models) == 0:
                return self.base_model(self.min_n_components)
            bestModel = max(models, key = lambda x: x[1])[0]
        return bestModel


# TODO: I'm not sure if I should use the self.base_model() function here,
# but since I need to split the sequences into folds I don't know how to
# do this without overly complicated overriden and restoring of class attributes.
class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        with np.errstate(divide='ignore'):
            models = dict()
            for n in range(self.min_n_components, self.max_n_components + 1):
                # Keep track of the score for n components.
                score = 0
                # Calculate the number of available folds.
                n_splits = min(len(self.sequences), 3)
                # If we don't have enough data to evaluate the model, we simply return
                # the simplest model, with the fewest number of components.
                if n_splits < 2:
                    return self.base_model(self.min_n_components)
                # Split data into TRAIN and TEST folds.
                split_method = KFold(n_splits)
                for trainIndices, testIndices in split_method.split(self.sequences):
                    # Train the GaussianHMM on TRAIN folds and n components.
                    xTrain, lengthsTrain = combine_sequences(trainIndices, self.sequences)
                    try:
                        model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                                random_state=self.random_state, verbose=False).fit(self.xTrain, self.lengthsTrain)
                        # Score the model based on the TEST data.
                        xTest, lengthsTest = combine_sequences(testIndices, self.sequences)
                        score += model.score(xTest, lengthsTest)
                    except:
                        # If we run into an error, we skip this number of components.
                        break
                models[n] = score / n_splits
            bestNumComponents = max(models, key=models.get)
        return self.base_model(bestNumComponents)
