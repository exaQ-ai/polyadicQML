"""Optimization module.

We use the update steps from `"Deep Learning - Goodfellow et al."`
"""


class Optimizer():
    def __init__(self, model, lr):
        self.model = model
        self.set_lr(lr)

    def step(self, X, y):
        """Update the parameters of the model based on input `X` and labels
        `y`.

        Parameters
        ----------
        X : array-like
            Design matrix
        y : vector-like
            Target vector
        """
        raise NotImplementedError

    def set_lr(self, lr):
        self.lr = lr


class SGD(Optimizer):
    def __init__(self, model, lr, momentum=0):
        super().__init__(model, lr)

        self.momentum = momentum
        self.v = 0

    def step(self, X, y):
        G = self.model.grad(X, y)
        self.v = self.momentum * self.v - self.lr * G

        self.model.params += self.v


OPTIM_METHODS = dict(
    sgd = SGD,
)
