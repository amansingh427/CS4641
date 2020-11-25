import numpy as np
import json
from sklearn.feature_extraction import text

class Naive_Bayes():

    def __init__(self): 
        ## NOTHING TODO
        # load emails
        x = open('emails.txt').read()
        emails = json.loads(x)
        
        # get previous spam emails (spam), non spam emails (not_spam), unclassified input mails (to_classify)
        spam = emails["spam"]
        not_spam = emails["not_spam"]
        to_classify = emails["to_classify"]
        
        # Number of emails
        n_spam = len(spam)
        n_not_spam = len(not_spam)
        n_to_classify = len(to_classify)
        
        ''' To ignore certain common words in English that might skew your model, we add them to the stop words 
         list below. You may want to experiment by choosing your own list of stop words, 
         but be sure to keep subject in this list at a minimum, as it appears in every email content.'''
        stop_words = text.ENGLISH_STOP_WORDS.union({'subject'})        
        
        # Form bag of words model using words used at least 10 times
        vectorizer = text.CountVectorizer(stop_words=stop_words,min_df=10)
        X = vectorizer.fit_transform(spam+not_spam+to_classify).toarray()
        
        # split word counts into separate matrices
        self.X_spam, self.X_not_spam, self.X_to_classify = X[:n_spam,:], X[n_spam:n_spam+n_not_spam,:], X[n_spam+n_not_spam:,:]
        
    def _likelihood_ratio(self, X_spam, X_not_spam):
        '''
        Args:
            X_spam: n_spam x d where n_spam is the number of spam emails,
                and d is the number of unique words that were there in all the emails
            X_not_spam: n_not_spam x d where n_not_spam is the number of good emails,
                and d is the number of unique words that were there in all the emails
        Return:
            ratio: 1 x d vector of the likelihood ratio of different words (spam/not_spam)
        '''
        
        '''
        Hints: 
            The value of X_spam[i][j] gives the number of times the jth word appears in the ith email
            Likelihood is P(words|class)
            Add 1 to avoid all zeros result
        '''
        ######### TODO - PASTE FUNCTION #########
        
        # estimate probability of each word in vocabulary being used in spam
        p_spam = (1 + X_spam.sum(axis=0)) / (sum(np.sum(a) for a in X_spam) + X_spam.shape[1])
        # estimate probability of each word in vocabulary being used in clean emails
        p_not_spam = (1 + X_not_spam.sum(axis=0)) / (sum(np.sum(a) for a in X_not_spam) + X_not_spam.shape[1])
        # compute ratio of these probabilities
        return p_spam / p_not_spam
    
    def _priors_ratio(self, X_spam, X_not_spam):
        '''
        Args:
            X_spam: n_spam x d where n_spam is the number of spam emails,
                and d is the number of unique words that were there in all the emails
            X_not_spam: n_not_spam x d where n_not_spam is the number of good emails,
                and d is the number of unique words that were there in all the emails
        Return:
            pr: prior ratio of (spam/not_spam)
        '''
        ######### TODO - PASTE FUNCTION #########
        
        # Compute prior probabilities
        return sum(np.sum(a) for a in X_spam) / sum(np.sum(a) for a in X_not_spam)
        
    def classify_spam(self, likelihood_ratio, pratio, X_to_classify):
        '''
        Args:
            likelihood_ratio: 1 x d vector of ratio of likelihoods of different words
            pratio: 1 x 1 number
            X_to_classify: bag of words representation of the unknown emails
        Return:
             resolved: 1 x K list, each entry is 'S' to indicate spam or 'NS' to indicate not spam. 
             K is the number of emails to classify
        '''
        ######### TODO - PASTE FUNCTION #########
        
        # Iterate over emails (for loop allowed to go through the unclassified emails)
        
        # Compute likelihood ratio for Naive Bayes model
        
        # if posterior is greater than 0.5 classify it as spam ('S')
        resolved = []
        for email in X_to_classify:
            if np.prod(likelihood_ratio ** email) * pratio > 0.5:
                resolved.append('S')
            else:
                resolved.append('NS')
        return resolved
    
    
if __name__ == '__main__':
    #### NOTHING TODO #####
    NB = Naive_Bayes()
    likelihood_ratio = NB._likelihood_ratio(NB.X_spam, NB.X_not_spam)
    pratio = NB._priors_ratio(NB.X_spam, NB.X_not_spam)
    resolved = NB.classify_spam(likelihood_ratio, pratio, NB.X_to_classify)
    print(resolved)