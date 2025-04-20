



# Test on simple data
def test_simple_data():
    # Generate some simple data
    X = torch.tensor(np.linspace(-3, 3, 100).reshape(-1, 1), dtype=torch.float64)
    y = X**4 - 4*X + torch.normal(0, 5, size=X.shape)
    
    # Create a feature map for quadratic features
    def square(x):
        return x**2
    
    class RandomFeatures:
        def __init__(self, n_features, activation=None):
            self.n_features = n_features
            self.u = None
            self.b = None
            self.activation = activation if activation is not None else lambda x: x
            
        def fit(self, X):
            self.u = torch.randn((X.size()[1], self.n_features), dtype=torch.float64)
            self.b = torch.rand(self.n_features, dtype=torch.float64)
            
        def transform(self, X):
            return self.activation(X @ self.u + self.b)
    
    # Use 10 random features with square activation 
    phi = RandomFeatures(n_features=10, activation=square)
    phi.fit(X)
    X_features = phi.transform(X)
    
    # Train model
    model = MyLinearRegression()
    opt = OverParameterizedLinearRegressionOptimizer(model)
    opt.fit(X_features, y.flatten())
    
    # Predict and visualize
    y_pred = model.predict(X_features)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='darkgrey', label='Data')
    plt.plot(X, y_pred, color='blue', label='Predictions')
    plt.title('Overparameterized Linear Regression')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()
    
    print(f"MSE loss: {model.loss(X_features, y.flatten()):.4f}")

# Run the test
test_simple_data()