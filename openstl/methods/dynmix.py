import torch
import torch.nn as nn
import math

from typing import List


def unfold(tens: torch.Tensor, mode, dims, align=2):
    """
    Unfolds tensor into matrix.

    Parameters
    ----------
    tens : torch.Tensor, tensor with shape == dims
    mode : int, which axis to move to the front
    dims : list, holds tensor shape

    Returns
    -------
    matrix : torch.Tensor, shape (dims[mode], prod(dims[/mode]))
    """
    # if mode == 0:
    #     return tens.reshape(dims[0], -1)
    # else:
    #     return np.moveaxis(tens, mode, 0).reshape(dims[mode], -1)
    return torch.moveaxis(tens, mode + align, align).reshape(list(tens.shape[:align]) + [dims[mode], -1])


def refold(vec: torch.Tensor, mode, dims, align=2):
    """
    Refolds vector into tensor.

    Parameters
    ----------
    vec : torch.Tensor, tensor with len == prod(dims)
    mode : int, which axis was unfolded along.
    dims : list, holds tensor shape

    Returns
    -------
    tens : torch.Tensor, tensor with shape == dims
    """

    tens = vec.reshape(list(vec.shape[:align]) + [dims[mode]] + [d for m, d in enumerate(dims) if m != mode])
    return torch.moveaxis(tens, align, mode + align)


def kron_vec_prod(As: List[torch.Tensor], vt, align=2):
    """
    Computes matrix-vector multiplication between
    matrix kron(As[0], As[1], ..., As[N]) and vector
    v without forming the full kronecker product.
    """
    dims = [A.shape[-1] for A in As]
    # vt = v.reshape([v.shape[0], v.shape[1]] + dims)
    for i, A in enumerate(As):
        # temp = A @ unfold(vt, i, dims)
        temp = torch.einsum('bnij,bnjk->bnik', A, unfold(vt, i, dims))
        vt = refold(temp, i, dims)
    return vt


class covariance(nn.Module):
    def __init__(self, x_dim, y_dim, pred_len, num_feature, device, n_components=1, train_L_x=False, train_L_y=False, train_L_t=False, train_L_f=False):
        super(covariance, self).__init__()

        self.n_components = n_components
        self.num_nodes = x_dim
        self.pred_len = pred_len
        self.delay = y_dim
        self.device = device
        self.num_feature = num_feature

        self._L_x = nn.Parameter(torch.zeros(n_components, x_dim, x_dim).detach(), requires_grad=train_L_x)
        self._L_y = nn.Parameter(torch.zeros(n_components, y_dim, y_dim).detach(), requires_grad=train_L_y)
        self._L_t = nn.Parameter(torch.zeros(n_components, pred_len, pred_len).detach(), requires_grad=train_L_t)
        self._L_f = nn.Parameter(torch.zeros(n_components, num_feature, num_feature).detach(), requires_grad=train_L_f)

        self.elu = torch.nn.ELU()
        self.act = lambda x: self.elu(x) + 1

    @property
    def L_x(self):
        return torch.tril(self._L_x)

    @property
    def L_y(self):
        return torch.tril(self._L_y)

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
        Ly = self.update_diagonal(self.L_y).to(self.device)
        Lt = self.update_diagonal(self.L_t).to(self.device)
        Lf = self.update_diagonal(self.L_f).to(self.device)
        return Lt, Lf, Lx, Ly


def bessel_k_approx(z, v, max_iter=100):
    """Compute an approximation of the modified Bessel function of the second kind of order v using the asymptotic expansion"""
    result = torch.zeros_like(z)
    for k in range(max_iter):
        term = 1.0
        for i in range(k):
            term *= ((4 * v**2 - (2 * i + 1)**2) / (8 * z))
        add_term = term / math.factorial(k)
        result += add_term
    result *= torch.exp(-z) * torch.sqrt(math.pi / (2 * z))
    return result


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

    def forward(self, pred: torch.Tensor, target: torch.Tensor, logw: torch.Tensor):
        L_list = self.covariance.get_L()

        if self.rho == 0:
            return self.masked_mse(pred, target)
        elif self.rho == 1:
            return self.get_nll(pred, target, L_list, logw)
        else:
            nll = self.get_nll(pred, target, L_list, logw)
            mse = self.masked_mse(pred, target)
            return self.rho * nll + (1-self.rho) * mse

    def get_nll(self, mu: torch.Tensor, target: torch.Tensor, L_list: List[torch.Tensor], logw: torch.Tensor):
        if self.nll == "MGD":
            return self.get_nll_MGD(mu, target, L_list, logw)
        # elif self.nll == "MLD":
        #     return self.get_nll_MLD(mu, target, L_list)
        # elif self.nll == "MLD_abs":
        #     return self.get_nll_MLD_abs(mu, target, L_list)
        # elif self.nll == "GAL":
        #     return self.get_nll_GAL(mu, target, L_list)
        else:
            raise NotImplementedError

    def get_nll_MGD(self, mu: torch.Tensor, target: torch.Tensor, L_list: List[torch.Tensor], logw: torch.Tensor):
        b, t, k, f, x, y = mu.shape

        target = target.unsqueeze(2)
        R_ext = (mu - target)
        R_ext = R_ext.permute(0, 2, 1, 3, 4, 5)  # B x K x T x F x W x H

        L_list = [l.transpose(-1, -2).unsqueeze(0).repeat(b, 1, 1, 1) for l in L_list]
        logdet = [l.diagonal(dim1=-1, dim2=-2).log().sum(-1) for l in L_list]

        # R_ext = R_ext.permute(0, 2, 1, 3, 4)
        # R_ext = R_ext.unsqueeze(1)
        L_x = kron_vec_prod(L_list, R_ext, align=2)
        mahabolis = L_x.pow(2).sum((-1, -2, -3, -4))

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
