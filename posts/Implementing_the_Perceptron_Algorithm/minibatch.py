import torch
from perceptron_model import LinearModel

class MiniBatchPerceptron(LinearModel):
    def loss(self, X, y):
        """
        Compute the misclassification rate.
        """

        y_ = 2*y - 1

        return (1.0 - ((y_ * self.score(X)) > 0).float()).mean()
    
    def grad(self, X, y):
        """
        Compute the gradient (update) for the perceptron algorithm.
        This version supports minibatch processing.
        """

        y_ = 2*y - 1
        
        scores = self.score(X)
        misclassified = (scores * y_ <= 0).float()
        weighted_updates = misclassified.unsqueeze(1) * y_.unsqueeze(1) * X
        
        batch_size = X.size(0)
        return -torch.sum(weighted_updates, dim=0) / batch_size

class MiniBatchPerceptronOptimizer:
    def __init__(self, model):
        self.model = model
    
    def step(self, X, y, learning_rate=1.0):
        """
        Compute one step of the minibatch perceptron update.
        """

        self.model.loss(X, y)
        grad_update = self.model.grad(X, y)
        self.model.w = self.model.w - learning_rate * grad_update