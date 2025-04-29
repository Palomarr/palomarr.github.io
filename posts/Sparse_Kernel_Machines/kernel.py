import torch

class KernelLogisticRegression:
    """
    Sparse Kernelized Logistic Regression with L1 regularization.
    
    This class implements logistic regression using the kernel trick, which allows
    for nonlinear decision boundaries while maintaining the computational simplicity
    of the linear model. The L1 regularization ensures sparsity in the weight vector.
    
    Parameters
        kernel : callable
            A function that computes the kernel between two sets of data points.
            Should have the signature: kernel(X_1, X_2, **kwargs) -> torch.Tensor
        lam : float, default=0.1
            Regularization strength. Larger values specify stronger regularization.
        **kwargs : dict
            Additional keyword arguments to pass to the kernel function.
    """
    
    def __init__(self, kernel, lam=0.1, **kwargs):
        self.kernel = kernel
        self.lam = lam
        self.kwargs = kwargs
        self.a = None
        self.X_train = None
        self.K_cache = None
        
    def sigmoid(self, x):
        """
        Compute the logistic sigmoid function.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
            
        Returns
        -------
        torch.Tensor
            Sigmoid of input tensor.
        """
        return 1 / (1 + torch.exp(-x))
    
    def score(self, X, recompute_kernel=False):
        """
        Compute scores for the input data using the kernel method.
        
        Parameters
        ----------
        X : torch.Tensor
            Feature matrix of shape (m, p).
        recompute_kernel : bool, default=False
            Whether to recompute the kernel matrix or use the cached version if available.
            
        Returns
        -------
        torch.Tensor
            Vector of scores of shape (m,).
        """
        if self.X_train is None or self.a is None:
            raise ValueError("Model has not been fitted yet.")
        
        # Compute or use cached kernel matrix
        if self.K_cache is None or recompute_kernel:
            K = self.kernel(X, self.X_train, **self.kwargs)
            self.K_cache = K
        else:
            K = self.K_cache
        
        # Compute the scores: s = K * a
        scores = K @ self.a
        
        return scores
    
    def predict_proba(self, X):
        """
        Predict class probabilities for the input data.
        
        Parameters
        ----------
        X : torch.Tensor
            Feature matrix of shape (m, p).
            
        Returns
        -------
        torch.Tensor
            Predicted probabilities of shape (m,).
        """
        scores = self.score(X)
        return self.sigmoid(scores)
    
    def predict(self, X):
        """
        Predict class labels for the input data.
        
        Parameters
        ----------
        X : torch.Tensor
            Feature matrix of shape (m, p).
            
        Returns
        -------
        torch.Tensor
            Predicted class labels (0 or 1) of shape (m,).
        """
        probs = self.predict_proba(X)
        return (probs >= 0.5).float()
    
    def loss(self, X, y):
        """
        Compute the binary cross-entropy loss with L1 regularization.
        
        Parameters
        ----------
        X : torch.Tensor
            Feature matrix of shape (m, p).
        y : torch.Tensor
            Target vector of shape (m,).
            
        Returns
        -------
        float
            The loss value.
        """
        m = X.size(0)
        scores = self.score(X)
        probs = self.sigmoid(scores)
        
        # Binary cross-entropy loss
        loss_bce = -torch.mean(y * torch.log(probs + 1e-10) + (1 - y) * torch.log(1 - probs + 1e-10))
        
        # L1 regularization
        l1_reg = self.lam * torch.norm(self.a, p=1)
        
        return loss_bce + l1_reg
    
    def gradient(self, X, y):
        """
        Compute the gradient of the loss function.
        
        Parameters
        ----------
        X : torch.Tensor
            Feature matrix of shape (m, p).
        y : torch.Tensor
            Target vector of shape (m,).
            
        Returns
        -------
        torch.Tensor
            Gradient of the loss with respect to the weight vector a.
        """
        m = X.size(0)
        K = self.kernel(X, self.X_train, **self.kwargs)
        scores = K @ self.a
        probs = self.sigmoid(scores)
        
        # Gradient of binary cross-entropy
        grad_bce = (1/m) * K.T @ (probs - y)
        
        # Gradient of L1 regularization (subgradient)
        grad_l1 = self.lam * torch.sign(self.a)
        
        return grad_bce + grad_l1
    
    def fit(self, X, y, m_epochs=1000, lr=0.01):
        """
        Fit the model to the training data using gradient descent.
        
        Parameters
        ----------
        X : torch.Tensor
            Feature matrix of shape (n, p).
        y : torch.Tensor
            Target vector of shape (n,).
        m_epochs : int, default=1000
            Maximum number of training epochs.
        lr : float, default=0.01
            Learning rate for gradient descent.
            
        Returns
        -------
        self : KernelLogisticRegression
            The fitted model.
        """
        # Store the training data for later use in prediction
        self.X_train = X
        
        # Initialize weight vector
        n = X.size(0)
        self.a = torch.zeros(n, dtype=torch.float32)
        
        # Gradient descent
        for epoch in range(m_epochs):
            # Compute gradient
            grad = self.gradient(X, y)
            
            # Update weights
            self.a -= lr * grad
        
        return self