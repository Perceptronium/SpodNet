import torch
from torch import nn
from torch import einsum


class Abs(nn.Module):
    def __init__(self):
        super(Abs, self).__init__()
        self.abs = torch.abs

    def forward(self, x):
        x = self.abs(x)
        return x


class UpdateTheta(nn.Module):
    """ Layer class : Updates every column/row/diagonal element of the input matrix. """

    def __init__(self, p, theta_12_generator=None, device=None):
        super().__init__()

        self.p = p
        self.device = device

        self.W = None
        self.S = None

        if theta_12_generator is None:
            raise NotImplementedError(
                "You need to provide a function to update columns.")

        elif theta_12_generator == 'UBG':
            self.theta_12_generator = UpdateTheta12UBG(
                p, self.device)

        elif theta_12_generator == 'PNP':
            self.theta_12_generator = UpdateTheta12PNP(
                p, self.device)

        elif theta_12_generator == 'E2E':
            self.theta_12_generator = UpdateTheta12E2E(
                p, self.device)
        else:
            raise NotImplementedError("Unrecognized column updater.")

        # Diagonal updater's g function
        self.alpha_learner = nn.Sequential(
            nn.Linear(3, 3, dtype=torch.float64), nn.ReLU(),
            nn.Linear(3, 3, dtype=torch.float64), nn.ReLU(),
            nn.Linear(3, 1, dtype=torch.float64), Abs()
        )

    def forward(self, Theta):
        """ A single layer update: update column/row/diagonal of all indices. """

        indices = torch.arange(self.p)

        for col in range(self.p):
            indices_minus_col = torch.cat([indices[:col], indices[col + 1:]])
            _11 = slice(
                None), indices_minus_col[:, None], indices_minus_col[None]
            _12 = slice(None), indices_minus_col, col
            _21 = slice(None), col, indices_minus_col
            _22 = slice(None), col, col

            # Blocks of W
            W_11 = self.W[_11]
            w_22 = self.W[_22]
            w_12 = self.W[_12]

            # Blocks of Theta
            theta_22 = Theta[_22]
            theta_12 = Theta[_12]
            inv_Theta_11 = W_11 - _a_outer(1/w_22, w_12, w_12)

            # Blocks of S
            s_12 = self.S[_12]
            s_22 = self.S[_22]

            # Compute theta_12_next
            theta_12_next = self.theta_12_generator(
                theta_12, s_12, w_12, inv_Theta_11, Theta, col)

            # Diagonal element update
            quad_prod = _quad_prod(inv_Theta_11, theta_12_next)

            alpha_learner_features = torch.cat(
                (quad_prod[:, None], s_22[:, None], theta_22[:, None]), -1)

            alpha = self.alpha_learner(alpha_learner_features).squeeze()

            theta_22_next = alpha + quad_prod

            # Update Theta
            Delta = torch.zeros_like(Theta)
            Delta[_22] = theta_22_next - theta_22
            Delta[_12] = theta_12_next - theta_12
            Delta[_21] = Delta[_12].clone()
            Theta = Theta + Delta

            # Update W
            w_22_next = 1.0 / (theta_22_next - quad_prod)

            w_12_next = einsum('b, bij, bj ->bi', -w_22_next,
                               inv_Theta_11, theta_12_next)

            self.W[_11] = (inv_Theta_11 +
                           _a_outer(1/w_22_next, w_12_next, w_12_next))

            self.W[_12] = w_12_next
            self.W[_21] = w_12_next
            self.W[_22] = w_22_next

            if torch.any(torch.isnan(self.W)):
                print("Found NaN value in W.")
                breakpoint()

        return Theta


class UpdateTheta12UBG(nn.Module):
    def __init__(self, p, device):
        super().__init__()
        self.p = p
        self.gamma_learner = nn.Sequential(
            nn.Linear(self.p-1, int((self.p-1)/2),
                      dtype=torch.float64), nn.ReLU(),
            nn.Linear(int((self.p-1)/2), 1, dtype=torch.float64), Abs()
        )

        self.st_param_learner = nn.Sequential(
            nn.Linear(1, 5, dtype=torch.float64), nn.ReLU(),
            nn.Linear(5, 1, dtype=torch.float64), Abs()
        )

        self.zero = torch.Tensor([0]).type(torch.float64).to(device)

    def forward(self, theta_12, s_12, w_12, inv_Theta_11, Theta, col):

        zeta = torch.tensor([1.])

        # Compute forward of the forward-backward
        gamma = self.gamma_learner(theta_12)
        z_12 = theta_12 - gamma * (s_12 - w_12)

        # Normalize the vector for stable training (details in paper)
        quad_prod = _quad_prod(inv_Theta_11, z_12)[:, None]
        z_12_normalized = (z_12 * torch.sqrt(zeta) /
                           torch.sqrt(quad_prod))

        # Infer Lambda for the ST operator
        batch_size = z_12_normalized.shape[0]
        z_12_normalized_r = z_12_normalized.reshape(batch_size, -1, 1)
        Lambda = self.st_param_learner(
            z_12_normalized_r).reshape(z_12_normalized.shape)

        # Compute backward of the forward-backward
        theta_12_next = (torch.sign(z_12_normalized) *
                         torch.maximum(self.zero, torch.abs(z_12_normalized) - Lambda))

        return theta_12_next


class UpdateTheta12PNP(nn.Module):
    def __init__(self, p, device):
        super().__init__()
        self.p = p

        self.gamma_learner = nn.Sequential(
            nn.Linear(self.p-1, int((self.p-1)/2),
                      dtype=torch.float64), nn.ReLU(),
            nn.Linear(int((self.p-1)/2), 1, dtype=torch.float64), Abs()
        )

        self.psi_learner = nn.Sequential(
            nn.Linear(self.p-1, self.p*2, dtype=torch.float64), nn.ReLU(),
            nn.Linear(self.p*2, self.p-1, dtype=torch.float64),
        )

        self.st_param_learner = nn.Sequential(
            nn.Linear(1, 5, dtype=torch.float64), nn.ReLU(),
            nn.Linear(5, 1, dtype=torch.float64), Abs()
        )

        self.zero = torch.Tensor([0]).type(torch.float64).to(device)

    def forward(self, theta_12, s_12, w_12, inv_Theta_11, Theta, col):
        zeta = torch.tensor([1.])

        # Compute forward of the forward-backward
        gamma = self.gamma_learner(theta_12)
        z_12 = theta_12 - gamma * (s_12 - w_12)

        # Infer the backward of the forward-backward
        psi_z_12 = self.psi_learner(z_12)

        # Normalize the vector for stable training (details in paper)
        quad_prod = _quad_prod(inv_Theta_11, psi_z_12)[:, None]
        psi_z_12_normalized = (psi_z_12 * torch.sqrt(zeta) /
                               torch.sqrt(quad_prod))

        # Infer Lambda for the ST operator
        batch_size = psi_z_12_normalized.shape[0]
        psi_z_12_normalized_r = psi_z_12_normalized.reshape(batch_size, -1, 1)
        Lambda = self.st_param_learner(
            psi_z_12_normalized_r).reshape(psi_z_12_normalized.shape) * 0.1

        theta_12_next = (torch.sign(psi_z_12_normalized) *
                         torch.maximum(self.zero, torch.abs(psi_z_12_normalized) - Lambda))

        return theta_12_next


class UpdateTheta12E2E(nn.Module):
    def __init__(self, p, device):
        super().__init__()
        self.p = p

        self.phi_learner = nn.Sequential(
            nn.Linear(self.p-1, self.p*10, dtype=torch.float64), nn.ReLU(),
            nn.Linear(self.p*10, self.p-1, dtype=torch.float64)
        )

        self.st_param_learner = nn.Sequential(
            nn.Linear(1, 5, dtype=torch.float64), nn.ReLU(),
            nn.Linear(5, 1, dtype=torch.float64), Abs()
        )

        self.st_params = nn.Parameter(torch.zeros(1) - 4).to(device)

        self.zero = torch.Tensor([0]).type(torch.float64).to(device)

    def forward(self, theta_12, s_12, w_12, inv_Theta_11, Theta, col):
        zeta = torch.tensor([1.])

        # Infer the whole forward-backward
        phi_z_12 = self.phi_learner(theta_12)

        # Normalize the vector for stable training (details in paper)
        quad_prod = _quad_prod(inv_Theta_11, phi_z_12)[:, None]
        phi_z_12_normalized = (phi_z_12 * torch.sqrt(zeta) /
                               torch.sqrt(quad_prod))

        # Infer Lambda for the ST operator
        batch_size = phi_z_12_normalized.shape[0]
        phi_z_12_normalized_r = phi_z_12_normalized.reshape(batch_size, -1, 1)
        Lambda = self.st_param_learner(
            phi_z_12_normalized_r).reshape(phi_z_12_normalized.shape) * 0.1

        theta_12_next = (torch.sign(phi_z_12_normalized) *
                         torch.maximum(self.zero, torch.abs(phi_z_12_normalized) - Lambda))

        return theta_12_next


def torch_ST(x, tau):
    """Perform element-wise soft-thresholding on a batch of matrices."""
    return torch.sign(x) * torch.maximum(torch.abs(x) - tau, torch.tensor(0.))


def _quad_prod(H, vec):
    """batch version of ``vec @ H @ vec``."""
    return einsum("bi,bij,bj->b", vec, H, vec)


def _a_outer(a, vec_1, vec_2):
    """batch version of ``a * outer(vec_1, vec_2)``."""
    return einsum("b,bi,bj->bij", a, vec_1, vec_2)


def get_off_diag(M):
    res = M.clone()
    res.diagonal(dim1=-1, dim2=-2).zero_()
    return res
