import numpy as np
from ModalAnalysis import ModalAnalysis as ma
import pandas as pd
import matplotlib.pyplot as plt

class FSO:
    def __init__(self, cost_fn, n_vars, Nf = 10, Nm = 10, ub=1, lb=0, S1_max = 100, L1 = 5, L2 = 5, S2_max = 153, verbose = False):
        self.cost_fn = cost_fn
        self.n_vars = n_vars
        self.Nf = Nf
        self.Nm = Nm
        self.ub = ub
        self.lb = lb
        self.F, self.Ff = self.initialize_population()
        self.S1_max = S1_max
        self.L1 = L1
        self.L2 = L2
        self.S2_max = S2_max
        self.verbose = verbose
        

    def initialize_population(self):
        pop=[]
        costs = []
        self.best_solution = None
        self.best_cost = np.inf
        self.X = []
        self.Xf = []
        for i in range(self.Nm):
            pop.append(self.lb+(self.ub - self.lb)*np.random.random((self.n_vars,self.Nf)))
            costs.append(np.zeros(shape=(self.Nf,)))
            for j in range(self.Nf):
                costs[i][j] = self.cost_fn(pop[i][:,j])
            
            min_idx = np.argmin(costs[i])
            self.Xf.append(costs[i][min_idx].copy())
            self.X.append(pop[i][:,min_idx].copy())

            if self.best_cost>costs[i][min_idx]:
                self.best_cost = costs[i][min_idx].copy()
                self.best_solution = pop[i][:,min_idx].copy()
        
        return (pop, costs)

    def run(self):
        log=[]
        for s in range(1,self.S1_max+1):
            if self.verbose :
                print("step {} : {}".format(s,self.best_cost))
            
            for l in range(1,self.L1+1):
                a=np.random.permutation(self.Nm)

                for m in range(self.Nm):
                    C1=-0.75+2.255*np.random.random((self.n_vars,self.Nf))
                    C2=-0.25+1.302*np.random.random((self.n_vars,self.Nf))
                    b=a[m]

                    Mx=np.tile(self.X[m].reshape(-1,1),(1,self.Nf))
                    My=np.tile(self.X[b].reshape(-1,1),(1,self.Nf))

                    self.F[m]=self.F[m]+C1*(Mx-self.F[m])+C2*(My-self.F[m])
                    self.F[m][self.F[m]<self.lb]=self.lb
                    
                    self.Ff[m]=np.apply_along_axis(self.cost_fn,0,self.F[m]).reshape(-1,)

                    min_idx=np.argmin(self.Ff[m])
                    self.Xf[m]=self.Ff[m][min_idx].copy()
                    self.X[m]=self.F[m][:,min_idx].copy()

                    if self.Xf[m]<self.best_cost:
                        self.best_cost=self.Xf[m].copy()
                        self.best_solution=self.X[m].copy().reshape(-1,)

            for l in range(self.L2):
                for m in range(self.Nm):
                    c3=-0.5+2.7*np.random.random((self.n_vars,))
                    self.X[m]=self.X[m]+c3*(self.best_solution-self.X[m])
                    self.X[m][self.X[m]<self.lb]=self.lb
                    self.Xf[m]=self.cost_fn(self.X[m])

                for m in range(self.Nm):
                    if self.Xf[m]<self.best_cost:
                        self.best_cost=self.Xf[m].copy()
                        self.best_solution=self.X[m].copy().reshape(-1,)
            
            log.append(self.best_cost.copy())
                
                
                
        
        for s in range(1,self.S2_max+1):
            a=np.random.permutation(self.Nm)
            for m in range(self.Nm):
                c4=1.4*np.random.random((self.n_vars,))
                b=a[m]
                self.X[m]=self.X[m]+c4*(self.best_solution-self.X[b])
                self.X[m][self.X[m]<self.lb]=self.lb

                self.Xf[m]=self.cost_fn(self.X[m].reshape(-1,))
                if self.Xf[m]<self.best_cost:
                    self.best_cost=self.Xf[m]
                    self.best_solution=self.X[m].copy().reshape(-1,)
        
        return log

save_folder='3D_1000_300_15_5_localized'

def main(iter):
    file_name = '3D-data.xlsx'
    dimension = int(file_name[0])
    elements = pd.read_excel(file_name, sheet_name='Sheet1').values[:, 1:]
    nodes = pd.read_excel(file_name, sheet_name='Sheet2').values[:, 1:]
    # arrested_dofs=np.array([0,1,42,43])
    arrested_dofs=np.arange(0,24)
    aa = ma(elements, nodes, dimension,arrested_dofs=arrested_dofs)
    M=aa.assembleMass()

    x_exp=np.zeros(len(elements))
    # x_exp[5]=0.35
    # x_exp[23]=0.20
    x_exp[15]=0.4   #localized
    # x_exp[10]=0.24

    K=aa.assembleStiffness(x_exp)
    w_exp, v_exp=aa.solve_eig(K,aa.M)
    
    num_modes=10

    w_exp=w_exp[:num_modes]
    v_exp=v_exp[:,:num_modes]
    F_exp=np.sum(v_exp*v_exp,axis=0)/(w_exp*w_exp)
    # print("w_exp",w_exp)

    def objective_function(x):
        K=aa.assembleStiffness(x)
        w, v = aa.solve_eig(K, aa.M)
        w=w[:num_modes]
        v=v[:,:num_modes]
        
        MAC=(np.sum((v*v_exp),axis=0)**2)/(np.sum(v*v,axis=0)*np.sum(v_exp*v_exp,axis=0))
        
        F=np.sum(v*v,axis=0)/(w*w)
        MACF=(np.sum(F*F_exp)**2)/(np.sum(F*F)*np.sum(F_exp*F_exp))

        MDLAC=(np.abs(w-w_exp)/w_exp)**2
        cost = np.sum(1-MAC)+np.sum(MDLAC)+np.sum(1-MACF)
        return cost
    
    print(objective_function(x_exp))

    optimizer = FSO(n_vars=len(elements),cost_fn=objective_function, S1_max=1000,S2_max=300, Nf=15, Nm=5,ub=1,lb=0, verbose = True)
    
    log=optimizer.run()

    plt.yscale('log')
    plt.plot(log,'r-')
    plt.ylabel('cost')
    plt.xlabel('iteration')
    plt.savefig(f'./{save_folder}/convergence_{iter}.png')
    plt.clf()

    print(optimizer.best_cost, optimizer.best_solution)
    return optimizer.best_solution



if __name__ == '__main__':
    def func(x):
        n=5
        return n*10+np.sum(x**2-10*np.cos(2*np.pi*x))

    optimizer = FSO(cost_fn=func, S1_max=50,S2_max=300, n_vars = 5, Nf=10, Nm=10,ub=5.12,lb=-5.12,verbose = True)
    log=np.array(optimizer.run())
    log[log<1e-15]=1e-15
    print(optimizer.best_cost, optimizer.best_solution)
    plt.yscale('log')
    plt.plot(log,'r-')
    plt.ylabel('cost')
    plt.xlabel('iteration')
    plt.savefig('./rastrigin_convergence_5D.png')
    plt.clf()

    # best_sols=[]

    # for i in range(1,6):
    #     best_sols.append(main(i).reshape(-1,))
    
    # np.savetxt(f'./{save_folder}/best_sols.csv',np.stack(best_sols),delimiter=',')