import torch

class LinearModel:

    def __init__(self):

        self.w = None

    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X.
        The formula for the ith entry of s is s[i] = <self.w, x[i]>.
        If self.w currently has value None, then it is necessary to first initialize self.w to a random value.

        ARGUMENTS:
        X, torch.Tensor: the feature matrix. X.size() == (n, p), where n is the number of data points and p is the number of features. This implementation always assumes that the final column of X is a constant column of 1s.

        RETURNS:
        s torch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w is None:

            self.w = torch.rand((X.size()[1]))
        # Compute s[i] = <self.w, x[i]> for all i
        return X @ self.w

    

    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith data point is either 0 or 1.

        ARGUMENTS:
        X, torch.Tensor: the feature matrix. X.size() == (n, p), where n is the number of data points and p is the number of features. This implementation always assumes that the final column of X is a constant column of 1s.

        RETURNS:
        y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """
        # Compute scores

        scores = self.score(X)

        # Convert scores to predictions (1.0 if score > 0)

        return (scores > 0).float()


class LogisticRegression(LinearModel):
    """
    Logistic Regression classifier using the binary cross-entropy loss.
    """

    def loss(self, X, y):
        """
        Compute the binary cross-entropy loss.

        Args:
            X (torch.Tensor): Feature matrix of shape (n, p)
            y (torch.Tensor): Target vector of shape (n,)
            
        Returns:
            torch.Tensor: The mean binary cross-entropy loss
        """
        # Compute sigmoid function
        sigmoid = torch.sigmoid(self.score(X))
        # Compute binary cross-entropy loss for each observation then return mean loss
        return (-y * torch.log(sigmoid) - (1 - y) * torch.log(1 - sigmoid)).mean()

    def grad(self, X, y):
        """
        Compute the gradient of the binary cross-entropy loss with respect to w.
        
        Args:
            X (torch.Tensor): Feature matrix of shape (n, p)
            y (torch.Tensor): Target vector of shape (n,)
            
        Returns:
            torch.Tensor: Gradient vector of shape (p,)
        """
        # Compute sigmoid function
        sigmoid = torch.sigmoid(self.score(X))
        n = X.size(0)
        # Calculate gradient
        return (1/n) * (sigmoid - y) @ X
    
class GradientDescentOptimizer:
    """
    Optimizer that implements gradient descent with momentum.
    """

    def __init__(self, model):
        """
        Initialize the optimizer with a model.
        
        Args:
            model: A model with a grad method that computes gradients
        """
        self.model = model
        if self.model.w is not None:
            self.w_prev = self.model.w.clone()
        else:
            self.w_prev = None
    
    def step(self, X, y, alpha, beta):
        """
        Perform one step of gradient descent with momentum.
        
        Args:
            X (torch.Tensor): Feature matrix
            y (torch.Tensor): Target vector
            alpha (float): Learning rate parameter
            beta (float): Momentum parameter
            
        Returns:
            None (updates model.w in-place)
        """
        # Initialize previous weight if needed
        if self.w_prev is None and self.model.w is not None:
            self.w_prev = self.model.w.clone()

        # Compute gradient
        grad = self.model.grad(X, y)

        # Store current weights temporarily
        w_current = self.model.w.clone()
        
        # Update weights using gradient descent with momentum
        self.model.w = self.model.w - alpha * grad + beta * (self.model.w - self.w_prev)
        
        # Update previous weights for next iteration
        self.w_prev = w_current