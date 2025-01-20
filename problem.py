import torch
import numpy as np

class ClassPb(object):
    """
    A class for the classification problem of the homework.

    Attributes:
        A (numpy.ndarray or torch.Tensor): Data matrix (features).
        y (numpy.ndarray or torch.Tensor): Data vector (labels).
        n (int): Number of data points (rows in A).
        d (int): Number of features (columns in A).
        lbda (float): Regularization parameter.
        reg (str): Regularization function type ('l1' for L1 regularization).
    """

    def __init__(self, A, y, lbda=0):
        """
        Initialize the classification problem.

        Args:
            A (numpy.ndarray): Input feature matrix of shape (n, d).
            y (numpy.ndarray): Input labels vector of length n.
            lbda (float, optional): Regularization parameter. Defaults to 0.
        """
        self.A = A
        self.y = y
        self.n, self.d = A.shape
        self.loss = self.obj_func
        self.lbda = lbda
        self.reg = None

    def convert_to_torch(self):
        """
        Convert data attributes A and y to PyTorch tensors with double precision.
        """
        self.A = torch.tensor(self.A, dtype=torch.float64)
        self.y = torch.tensor(self.y, dtype=torch.float64)
        return self

    def init_x(self):
        """
        Initialize the weight vector x with random values.
        """
        self.x = torch.zeros(self.d, dtype=torch.float64) #torch.rand(self.d, dtype=torch.float64)

    def obj_func_i(self, x, i):
        """
        Compute the value of the i-th component of the objective function.

        Args:
            x (torch.Tensor): Weight vector of shape (d,).
            i (int): Index of the data point.

        Returns:
            torch.Tensor: Scalar value of the i-th component of the objective function.
        """
        z = self.A[i].T @ x
        sigma = 1 / (1 + torch.exp(-z))  # Sigmoid function
        if self.reg == 'l1':
            return (self.y[i] - sigma) ** 2 + self.lbda * torch.norm(x, 1)
        return (self.y[i] - sigma) ** 2

    def obj_func(self, x):
        """
        Compute the value of the full objective function (mean squared error + regularization).

        Args:
            x (torch.Tensor): Weight vector of shape (d,).

        Returns:
            torch.Tensor: Scalar value of the objective function.
        """
        z = self.A @ x
        sigma = 1 / (1 + torch.exp(-z))  # Sigmoid function
        if self.reg == 'l1':
            return torch.mean((self.y - sigma) ** 2) + self.lbda * torch.norm(x, 1)
        return torch.mean((self.y - sigma) ** 2)

    def grad_explicit(self, x, i):
        """
        Compute the explicit gradient of the i-th component of the objective function.

        Args:
            x (torch.Tensor): Weight vector of shape (d,).
            i (int): Index of the data point.

        Returns:
            torch.Tensor: Gradient vector of shape (d,).
        """
        z = torch.clamp(self.A[i].T @ x, min=-50, max=50)  # Prevent overflow in exponential
        sigma = 1 / (1 + torch.exp(-z))  # Sigmoid function
        explicit_grad = -2 * (self.y[i] - sigma) * sigma * (1 - sigma) * self.A[i]
        if self.reg == 'l1':
            reg_grad = self.lbda * torch.sign(x)
            return explicit_grad + reg_grad
        return explicit_grad

    def lipgrad(self):
        """
        Compute the Lipschitz constant for the gradient of the loss function.

        Returns:
            torch.Tensor: Lipschitz constant.
        """
        L = torch.norm(self.A, p=2) ** 2 / (2. * self.n) + self.lbda
        return L
