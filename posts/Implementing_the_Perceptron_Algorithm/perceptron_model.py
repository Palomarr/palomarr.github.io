import torch

class LinearModel:
    def __init__(self):
        self.w = None
    
    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X.
        The formula for the ith entry of s is s[i] = <self.w, x[i]>.
        """
        
        if self.w is None:
            self.w = torch.rand((X.size()[1]))
        
        return X @ self.w
    
    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X.
        """

        scores = self.score(X)
        
        return (scores > 0).float()

class Perceptron(LinearModel):
    def loss(self, X, y):
        """
        Compute the misclassification rate.
        """

        y_ = 2*y - 1
        
        return (1.0 - ((y_ * self.score(X)) > 0).float()).mean()
    
    def grad(self, X, y):
        """
        Compute the gradient for the perceptron algorithm.
        For a single data point, returns -1[si(2yi-1) < 0]yi*xi.
        """
        y_ = 2*y - 1

        scores = self.score(X)
        misclassified = (scores * y_ <= 0).float()

        return -torch.sum(misclassified.unsqueeze(1) * y_.unsqueeze(1) * X, dim=0)

class PerceptronOptimizer:
    def __init__(self, model):
        self.model = model
    
    def step(self, X, y):
        """
        Compute one step of the perceptron update using the feature matrix X
        and target vector y.
        """

        self.model.loss(X, y)
        self.model.w = self.model.w - self.model.grad(X, y)
