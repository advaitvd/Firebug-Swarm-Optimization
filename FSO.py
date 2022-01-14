import numpy as np
from ModalAnalysis import ModalAnalysis as ma

class FSO:
    def __init__(self, cost_fn, n_vars, Nf = 5, Nm = 5, ub=1, lb=0, max_iterations = 100):
        self.cost_fn = cost_fn
        self.n_vars = n_vars
        self.Nf = Nf
        self.Nm = Nm
        self.ub = ub
        self.lb = lb
        self.F, self.Ff = self.initialize_population()
        self.max_iterations = max_iterations

    def initialize_population(self):
        arr = np.zeros((self.Nm,self.Nf,self.n_vars))
        costs = np.zeros((self.Nm,self.Nf))
        self.best_solution = None
        self.best_cost = np.inf
        self.X = np.zeros((self.Nm,self.n_vars))
        self.Xf = np.zeros((self.Nm,))
        for i in range(self.Nm):
            arr[i,:,:] = (self.lb+(self.ub - self.lb)*np.random.random((self.Nf, self.n_vars)))
            for j in range(self.Nf):
                costs[i, j] = self.cost_fn(arr[i,j,:])
            
            min_idx = np.argmin(costs[i])
            self.Xf[i] = costs[i,min_idx]
            self.X[i] = arr[i, min_idx, :].copy()

            if self.best_cost>costs[i, min_idx]:
                self.best_cost = costs[i, min_idx]
                self.best_solution = arr[i, min_idx, :].copy()
        
        return (arr, costs)

    def run(self):
        for step in range(1,self.max_iterations):
            pass


if __name__ == '__main__':
    def cost_fn(X):
        return np.sum((X-0.5)**2)

    optimizer = FSO(cost_fn=cost_fn, n_vars = 5, Nf=3, Nm=2)

    # print(optimizer.X, optimizer.best_solution)