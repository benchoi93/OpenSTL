import numpy as np

import torch
import torch.nn as nn
import math

from typing import List

from .dynmix import unfold, refold, kron_vec_prod


def matern_covariance(d, sigma, rho):
    term1 = 1 + (torch.sqrt(torch.tensor(5.)) * d) / rho
    term2 = (5 * d**2) / (3 * rho**2)
    term3 = torch.exp(-torch.sqrt(torch.tensor(5.)) * d / rho)
    return sigma**2 * (term1 + term2) * term3

def generate_matern_covariance_matrix(W, H, sigma, rho):
    # Generate all coordinates for the grid
    x_coords = torch.arange(0, W).float()
    y_coords = torch.arange(0, H).float()
    X, Y = torch.meshgrid(x_coords, y_coords)
    
    # Flatten and stack coordinates to create 2D point tensor
    points = torch.stack((X.flatten(), Y.flatten()), dim=1)
    
    # Compute pairwise squared differences
    diff = points.unsqueeze(1) - points.unsqueeze(0)
    squared_distances = (diff**2).sum(dim=2)
    
    # Compute actual pairwise distances
    distances = torch.sqrt(squared_distances)
    
    # Compute the covariance matrix using Matérn function
    C = matern_covariance(distances, sigma, rho)
    
    return C

# W = 3
# H = 3
# sigma = torch.tensor(1.).float()
# rho = torch.tensor(1.).float()

# C = generate_matern_covariance_matrix(W, H, sigma, rho)
# print(C)

class covariance(nn.Module):
    def __init__(self, x_dim, y_dim, pred_len, num_feature, device, n_components=1, train_L_x=False, train_L_y=False, train_L_t=False, train_L_f=False):
        super(covariance, self).__init__()

        self.n_components = n_components
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.pred_len = pred_len
        # self.delay = y_dim
        self.device = device
        self.num_feature = num_feature

        # Amat, Dmat, self.Lmat = generate_matrices(x_dim, y_dim)
        # self.I = torch.eye(x_dim*y_dim).unsqueeze(0).repeat(n_components, 1, 1)
        # self._L_x = nn.Parameter(torch.FloatTensor(self.Lmat).unsqueeze(0).repeat(n_components, 1, 1).detach() , requires_grad=False)
        # self._alpha = nn.Parameter(torch.rand(n_components).unsqueeze(-1).unsqueeze(-1).detach(), requires_grad=train_L_x)

        x_coords = torch.arange(0, x_dim).float()
        y_coords = torch.arange(0, y_dim).float()
        X, Y = torch.meshgrid(x_coords, y_coords)
        
        # Flatten and stack coordinates to create 2D point tensor
        points = torch.stack((X.flatten(), Y.flatten()), dim=1)
        
        # Compute pairwise squared differences
        diff = points.unsqueeze(1) - points.unsqueeze(0)
        squared_distances = (diff**2).sum(dim=2)
        
        # Compute actual pairwise distances
        self.distances = torch.sqrt(squared_distances)
        
        self._rho = nn.Parameter(torch.ones(n_components).detach(), requires_grad=train_L_x)

        self._L_x = nn.Parameter(torch.zeros(n_components, x_dim*y_dim, x_dim*y_dim).detach(), requires_grad=train_L_x)
        self._L_t = nn.Parameter(torch.zeros(n_components, pred_len, pred_len).detach(), requires_grad=train_L_t)
        self._L_f = nn.Parameter(torch.zeros(n_components, num_feature, num_feature).detach(), requires_grad=train_L_f)

        self.elu = torch.nn.ELU()
        self.act = lambda x: self.elu(x) + 1

    @property
    def L_x(self):
        return torch.tril(self._L_x, -1)
        # # return torch.linalg.cholesky(self.I + self._alpha * self._L_x)
        # return torch.linalg.cholesky( torch.stack([matern_covariance(self.distances, 1, r1) for r1 in self._rho] , 0) )

    @property
    def L_t(self):
        return torch.tril(self._L_t)

    @property
    def L_f(self):
        return torch.tril(self._L_f)

    def update_diagonal(self, L):
        N, D, _ = L.shape
        L[:, torch.arange(D), torch.arange(D)] = self.act(L[:, torch.arange(D), torch.arange(D)])
        return L

    def get_L(self) -> List[torch.Tensor]:
        Lx = self.update_diagonal(self.L_x).to(self.device)
        # Lx = self.L_x.to(self.device)
        Lt = self.update_diagonal(self.L_t).to(self.device)
        Lf = self.update_diagonal(self.L_f).to(self.device)
        return Lt, Lf, Lx


class better_loss(nn.Module):
    def __init__(self, in_shape, rho=1, n_components=3, det="mse", nll="MGD",
                 train_L_x=False, train_L_y=False, train_L_t=False, train_L_f=False):
        super(better_loss, self).__init__()
        self.pred_len, self.num_feature, self.xdim, self.ydim = in_shape

        self.rho = rho
        self.det = det
        self.nll = nll
        self.n_components = n_components

        self.train_L_x = train_L_x
        self.train_L_y = train_L_y
        self.train_L_t = train_L_t
        self.train_L_f = train_L_f

        self.covariance = covariance(
            x_dim=self.xdim,
            y_dim=self.ydim,
            pred_len=self.pred_len,
            num_feature=self.num_feature,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            n_components=n_components,
            train_L_x=train_L_x,
            train_L_y=train_L_y,
            train_L_t=train_L_t,
            train_L_f=train_L_f
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor, logw: torch.Tensor,sigma: torch.Tensor):
        L_list = self.covariance.get_L()

        if self.rho == 0:
            return self.masked_mse(pred, target)
        elif self.rho == 1:
            return self.get_nll(pred, target, L_list, logw,sigma)
        else:
            nll = self.get_nll(pred, target, L_list, logw,sigma)
            mse = self.masked_mse(pred, target)
            return self.rho * nll + (1-self.rho) * mse

    def get_nll(self, mu: torch.Tensor, target: torch.Tensor, L_list: List[torch.Tensor], logw: torch.Tensor, sigma: torch.Tensor):
        if self.nll == "MGD":
            return self.get_nll_MGD(mu, target, L_list, logw, sigma)
        # elif self.nll == "MLD":
        #     return self.get_nll_MLD(mu, target, L_list)
        # elif self.nll == "MLD_abs":
        #     return self.get_nll_MLD_abs(mu, target, L_list)
        # elif self.nll == "GAL":
        #     return self.get_nll_GAL(mu, target, L_list)
        else:
            raise NotImplementedError

    def get_nll_MGD(self, mu: torch.Tensor, target: torch.Tensor, L_list: List[torch.Tensor], logw: torch.Tensor, sigma: torch.Tensor):
        b, t, k, f, x, y = mu.shape

        L_list = [l.transpose(-1, -2).unsqueeze(0).repeat(b, 1, 1, 1) for l in L_list]
        sigma = torch.nn.functional.elu(sigma) + 1 + 1e-6
        sigma = sigma.permute(0,3,1,2).reshape(b,k,x*y)
        # (covariance)  = (correlation coefficient) * sigma (variance)
        # L_list[2] is cholesky factor of correlation matrix
        # compute covariance from L_list[2] and sigma
        # L_list[2] 's size is [B, K, N, N] 
        # sigma 's size is [B, K, N]
        L_list[2] = L_list[2] * sigma.unsqueeze(-1)

        target = target.unsqueeze(2)
        # mu = mu[:,:,:1].repeat(1,1,k,1,1,1)
        R_ext = (mu - target)
        R_ext = R_ext.permute(0, 2, 1, 3, 4, 5)  # B x K x T x F x W x H

        R_ext = R_ext.reshape(b, k, t, f, x*y)

        logdet = [l.diagonal(dim1=-1, dim2=-2).log().sum(-1) for l in L_list]

        L_x = kron_vec_prod(L_list, R_ext, align=2)
        mahabolis = L_x.pow(2).sum((-1, -2, -3))

        tfxy = t*f*x*y
        logdet = sum([tfxy*ll/L_list[i].shape[-1] for i, ll in enumerate(logdet)])

        nll = -tfxy/2 * math.log(2*math.pi) - 0.5 * mahabolis + logdet + logw
        nll = - torch.logsumexp(nll, dim=1)

        return nll.mean()

    def masked_mse(self, pred: torch.Tensor, target: torch.Tensor):
        mask = target != 0
        mask = mask.float()
        mask /= torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

        pred = pred.unsqueeze(1)

        if self.det == "mse":
            mse_loss = (pred-target) ** 2
        elif self.det == "mae":
            mse_loss = torch.abs(pred-target)
        else:
            raise NotImplementedError

        mse_loss = mse_loss * mask
        mse_loss = torch.where(torch.isnan(mse_loss), torch.zeros_like(mse_loss), mse_loss)
        mse_loss = torch.mean(mse_loss)
        return mse_loss


def get_adjacency_matrix(W, H):
    n = W * H
    A = np.zeros((n, n), dtype=int)

    for i in range(H):
        for j in range(W):
            node = i * W + j  # Convert 2D index to 1D

            # Check and set for left neighbor
            if j > 0:
                left = i * W + (j - 1)
                A[node][left] = 1

            # Check and set for right neighbor
            if j < W - 1:
                right = i * W + (j + 1)
                A[node][right] = 1

            # Check and set for top neighbor
            if i > 0:
                top = (i - 1) * W + j
                A[node][top] = 1

            # Check and set for bottom neighbor
            if i < H - 1:
                bottom = (i + 1) * W + j
                A[node][bottom] = 1

    return A


def get_degree_matrix(A):
    return np.diag(A.sum(axis=1))


def get_laplacian_matrix(A, D):
    return D - A


def generate_matrices(W, H):
    A = get_adjacency_matrix(W, H)
    D = get_degree_matrix(A)
    L = get_laplacian_matrix(A, D)

    return A, D, L


# W = 3  # For example
# H = 3  # For example
# A, D, L = generate_matrices(W, H)
# print("Adjacency Matrix:\n", A)
# print("Degree Matrix:\n", D)
# print("Laplacian Matrix:\n", L)
