from scipy.optimize import minimize
import numpy as np

class LogScoreModel:
    """ Abstract Model class for LogScoring """
    def __init__(self):
        self._params = []
    #Parameters of the model. The property makes it possible to compute additional steps before feeding the parameters to the user. 
    #The fitting is made on _params and the users will use params directly. It's an additional abstraction layer.
    @property
    def params(self):
        return self._params
    #Fit the model on the the loss (requires the definition of the forward pass)
    def fit(self,x,y):
        self._params = minimize(lambda params : self._loss(x,y,params),self._params,method='BFGS',tol=10**-20).x
    #Forward pass to be defined by subclassing
    def _forward(self,x,params):
        raise NotImplementedError
    def forward(self,x):
        return self._forward(x,self.params)
    def __call__(self,x):
        return self.forward(x)
    #Loss function (default is MSE but you can overload it if needed)
    def _loss(self,x,y,params):
        r = np.mean( (y-self._forward(x,params))**2 )
        return r
    def loss(self,x,y):
        return self._loss(x,y,self._params)

class ExponentialModel(LogScoreModel):
    """ Model corresponding to f(x) = -\alpha*(1-exp(-\beta*x)) """
    def __init__(self):
        self._params = [1,1] #Alpha beta
    def _forward(self,x,params):
        #Unpack parameters
        alpha,beta = params
        #Apply formula
        pred = -alpha*(1-np.exp(-beta*x))
        #Return prediction
        return pred