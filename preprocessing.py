import pandas as pd
import numpy as np

class preprocessing:

    @staticmethod
    def PCA(data,dimensions=2):
        X = data.iloc[:, :-1].values.T
        cov_matrix = data.iloc[:, :-1].cov().values

        eig_val,eig_vec=np.linalg.eig(cov_matrix)
        eig_pairs=[(np.abs(eig_val[i]), eig_vec[:,i])for i in range(len(eig_val))]

        #sort decreasing
        eig_pairs=sorted(eig_pairs,key=lambda x: x[0], reverse=True)

        w_matrix=np.zeros((len(eig_val),dimensions))
        for i in range(dimensions):
            w_matrix[:,i]=eig_pairs[i][1]

        new_x=np.dot(w_matrix.T,X).T
        df=pd.DataFrame(data=np.c_[new_x,data.iloc[:,-1].values])

        return df

    @staticmethod
    def FDA(data,dimensions=2):

        X = data.iloc[:, :-1].values
        y= data.iloc[:, -1].values
        total_mu = data.iloc[:, :-1].mean().values
        mu_per_class = np.array(data.groupby(data.columns[-1]).mean())

        d=X.shape[1]
        vals,counts=np.unique(data[data.columns[-1]] ,return_counts=True)

        S_W = np.zeros((d, d))
        for label,mv in zip(vals,mu_per_class):
            class_sc_mat = np.zeros((d, d))  # scatter matrix for every class
            for row in X[y==label]:
                row, mv = row.reshape(d, 1), mv.reshape(d, 1)  # make column vectors
                class_sc_mat += (row - mv).dot((row - mv).T)
            S_W += class_sc_mat  # sum class scatter matrices
        # print('within-class Scatter Matrix:\n', S_W)

        S_B=np.zeros((d,d))
        for i,mv in zip(range(len(vals)),mu_per_class):
            S_B+=counts[i]*(np.outer(mu_per_class[i]-total_mu,(mu_per_class[i]-total_mu).T))
        # print('between-class Scatter Matrix:\n',S_B)

        eig_vals,eig_vecs=np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

        eig_pairs=[(np.abs(eig_vals[i]), eig_vecs[:,i].real)for i in range(len(eig_vals))]
        eig_pairs=sorted(eig_pairs,key=lambda x: x[0], reverse=True)

        # w_marix = np.array([eig_pairs[i][1] for i in range(dimensions)])
        # print(w_marix.shape)

        w_matrix = np.zeros((len(eig_vals), dimensions))
        for i in range(dimensions):
            w_matrix[:, i] = eig_pairs[i][1]

        new_x =np.dot(X, w_matrix)
        return  pd.DataFrame(data=np.c_[new_x,data.iloc[:,-1].values])










