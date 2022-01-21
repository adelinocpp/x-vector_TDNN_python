# -*- coding: utf-8 -*-
import numpy as np
from sklearn.decomposition import PCA
import scipy
from graphviz import Digraph
from scipy.stats import multivariate_normal as gaussian
# -----------------------------------------------------------------------------
class bplda:
    def __init__(self, niter = 500):
        self.pca = None
        self.n_principal_components = 0;
        self.K = 0;
        self.F = 0;
        self.N = 0;
        self.m = 0;
        
    # ------------------------------------------------------------------------    
    def fit(self,X,y):
        self.K = len(np.unique(y))
        self.F = X.shape[0]
        self.N = X.shape[1]
        self.m = np.mean(X, axis=1)
        self.n_principal_components = self.K - 1;
        self.m.shape == (self.F,1)
        
        if not (X.shape[1] == y.shape[0]):
            print('PLDA error: oh dear! number of data samples should match the number of labels!')
            return     
        
        
        m_ks, sigma_ks, n_ks = [],[],[]
        for k in np.unique(y):
            # Get only the data associated with class k
            X_k = X[:,y==k]
        
            # Compute the mean, number of samples, and class covariance
            m_k = np.mean(X_k,axis=1)
            n_k = len(X_k)
            sigma_k = np.cov(X_k)
        
            # Append them all
            m_ks.append(m_k)
            n_ks.append(n_k)
            sigma_ks.append(sigma_k)
        
        m_ks = np.array(m_ks)
        n_ks = np.array(n_ks)
        sigma_ks = np.array(sigma_ks)
    
        assert m_ks.shape == (self.K,self.F)
        assert n_ks.shape == (self.K,)
        assert sigma_ks.shape == (self.K,self.F,self.F)
    
        S_b =  ((m_ks - self.m).T * n_ks/self.N)  @ (m_ks - self.m)
        S_w = np.sum(sigma_ks * ((n_ks-1)/self.N).reshape(-1,1,1), axis=0)
    
        matrix_rank = np.linalg.matrix_rank(S_w)
        
        if (self.F != matrix_rank):
            self.pca = PCA(n_components = matrix_rank)
            self.pca.fit(X.T)
        
        X_pca = self.__transform_from_D_to_X(X)
        print("Shape of X_pca =",X_pca.shape)

        self.m_pca = X_pca.mean(axis=1)
        n = self.N/self.K
        S_b, S_w = self.__compute_Sb_Sw(X_pca,y)

        # Compute W
        eigvals, eigvecs = scipy.linalg.eigh(S_b, S_w)
        W = eigvecs

        # Compute Lambdas
        Lambda_b = W.T @ S_b @ W
        Lambda_w = W.T @ S_w @ W

        # Compute A
        self.A = np.linalg.inv(W.T) * (n / (n-1) * np.diag(Lambda_w))**0.5
        print("Shape of A:", self.A.shape)
        # Compute Psi
        diag_Lambda_w = Lambda_w.diagonal()
        diag_Lambda_b = Lambda_b.diagonal()

        Psi = (n - 1)*(diag_Lambda_b/diag_Lambda_w) - (1/n)
        Psi[ Psi <= 0 ] = 0
        Psi = np.diag(Psi)
        u = self.__transform_from_X_to_U(X_pca)
        print("Shape of U:", u.shape)
        # Compute the relevant dimensions of Psi
        relevant_dims = np.squeeze(np.argwhere(Psi.diagonal() != 0))
        if relevant_dims.ndim == 0:
            relevant_dims = relevant_dims.reshape(1,)

        U_model = self.__transform_from_U_to_Umodel(X_pca,relevant_dims)
        print("Shape of U_model:", U_model.shape)
    
        prior_params = {
            "mean": np.zeros(relevant_dims),
            "cov": np.diag(Psi)[relevant_dims]
        }
        
        
        dot = Digraph()
        dot.node('v',"v")
        dot.node('1',"u1")
        dot.node('2',"u2")
        dot.node("a","x1")
        dot.node("b","x2")
        dot.edges(['v1',"v2","1a","2b"])
        dot
        
        posterior_params = {}
        for k in np.unique(y):
            u_model_k = U_model[:,y==k]
            n_k = u_model_k.shape[1]
            cov = prior_params["cov"] / (1 + n_k * prior_params["cov"])
            mean = np.sum(u_model_k, axis=1) * cov
            posterior_params[k] = {"mean": mean, "cov":cov}
        
        post_pred_params = posterior_params.copy()
        for k,params in post_pred_params.items():
            params["cov"] += 1
        
            
        log_prob_post = []
        for k, param in post_pred_params.items():
            mean,cov = param["mean"], param["cov"]
            log_probs_k = gaussian(mean,np.diag(cov)).logpdf(U_model.T)
            log_prob_post.append(log_probs_k.T)
        log_prob_post = np.array(log_prob_post).T
        
        normalize = True
        if normalize:
            logsumexp = np.log(np.sum(np.exp(log_prob_post),axis=-1))
            log_probs = log_prob_post - logsumexp[..., None]
        else:
            log_probs = log_prob_post
    
        categories = np.array([k for k in post_pred_params.keys()])
        predictions = categories[np.argmax(log_probs,axis=-1)]

        print("I still here!",predictions)
    # ------------------------------------------------------------------------    
    def __compute_Sb_Sw(self,X,y):
        K = len(np.unique(y))
        m = np.mean(X,axis=1)
        F,N = X.shape
        m_ks, sigma_ks, n_ks = [],[],[]
        for k in np.unique(y):
            # Get only the data associated with class k
            X_k = X[:,y==k]
        
            # Compute the mean, number of samples, and class covariance
            m_k = np.mean(X_k,axis=1)
            n_k = len(X_k)
            sigma_k = np.cov(X_k)
        
            # Append them all
            m_ks.append(m_k)
            n_ks.append(n_k)
            sigma_ks.append(sigma_k)
        
        m_ks = np.array(m_ks)
        n_ks = np.array(n_ks)
        sigma_ks = np.array(sigma_ks)
    
        assert m_ks.shape == (K,F)
        assert n_ks.shape == (K,)
        assert sigma_ks.shape == (K,F,F)
    
        S_b =  ((m_ks - m).T * n_ks/N)  @ (m_ks - m)
        S_w = np.sum(sigma_ks * ((n_ks-1)/N).reshape(-1,1,1), axis=0)
        return S_b, S_w
    # ------------------------------------------------------------------------    
    def __transform_from_D_to_X(self,x):
        if (self.pca is not None):
            return self.pca.transform(x.T).T
        else:
            return x
    # ------------------------------------------------------------------------    
    def __transform_from_X_to_D(self,x):
        if (self.pca is not None):
            return self.pca.inverse_transform(x.T).T
        else:
            return x
    # ------------------------------------------------------------------------    
    def __transform_from_X_to_U(self,x_pca):
        return ((x_pca.T-self.m_pca) @ np.linalg.inv(self.A)).T
    # ------------------------------------------------------------------------    
    def __transform_from_U_to_X(self,u):
        return (self.m_pca + (u.T @ self.A.T)).T
    
    # ------------------------------------------------------------------------    
    def __transform_from_U_to_Umodel(self,x,dims):
        u_model = x[dims,...]
        return u_model 
    # ------------------------------------------------------------------------    
    def __transform_from_Umodel_to_U(self,x,dims,u_dim):
        shape = (*x.shape[-1:], u_dim)
        u = np.zeros(shape)
        u[dims, ...] = x
        return u
    # ------------------------------------------------------------------------    
    def score(self,Xp,Xq):
        return 0