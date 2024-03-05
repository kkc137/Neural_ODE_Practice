import torch
from torch import nn
from torchdiffeq import odeint


class Encoder(nn.Module):
    def __init__(self, input_dim):
        super(Encoder, self).__init__()
        self.flatten = nn.Flatten()  # ？
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),  # try different combinations of activation functions.
            # non-linearity of ReLU might not be strong, which might affect the fitting effect of the data
            #Consider adjusting the types of activation functions during the training process
            nn.Linear(64, 2)  # output ：recovery rate, infection rate
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

# encoder = Encoder()
# encoded_params = encoder(input_data)
# decoder = Decoder()
# sir_solution = decoder(encoded_params)

def sir_model(t, y, beta, gamma):
    S, I, R = y.unbind()
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return torch.stack([dSdt, dIdt, dRdt])

def solve_sir(beta, gamma, S0, I0, R0, t):
    y0 = torch.tensor([S0, I0, R0], dtype=torch.float32)
    solution = odeint(lambda t, y: sir_model(t, y, beta, gamma), y0, t, method='dopri5')
    return solution


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Decoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.network(x)


# combine neural network
class SIRnet(nn.Module):
    def __init__(self, input_dim, output_dim, S0, I0, R0, time_tensor):
        super(SIRnet, self).__init__()
        self.encoder = Encoder(input_dim)
        self.decoder = Decoder(3, output_dim)  # Assuming the SIR model outputs 3 values (S, I, R) for each time step.
        # Storing initial conditions and time tensor for SIR model.
        self.S0 = S0
        self.I0 = I0
        self.R0 = R0
        self.time_tensor = time_tensor

    def forward(self, x):
        # Encode the input to get the parameters for the SIR model.
        encoded_params = self.encoder(x)
        # Assuming the first two outputs of the encoder are beta and gamma.
        beta, gamma = encoded_params[:, 0], encoded_params[:, 1]
        # Solve the SIR model with the encoded parameters.
        sir_solution = solve_sir(beta, gamma, self.S0, self.I0, self.R0, self.time_tensor)
        # Decode the solution of the SIR model.
        output = self.decoder(sir_solution)
        return output




# m, n, k = 10, 10, 10  # based on reality， it is a assumption
# input_dim = 3 * m * n * k
# output_dim = input_dim # fixed two dim
#
# # initialized model
# encoder = Encoder(input_dim)
# decoder = Decoder(3 * k, output_dim)  # output (3, k)
#
# # example data
# x = torch.randn(1, 3, m, n, k)  #
#
# # 编码
# encoded_params = encoder(x)
# beta, gamma = encoded_params[0][0], encoded_params[0][1]
#
# # SIR模型求解
# t = torch.linspace(0, 100, steps=k)  # 时间点
# S0, I0, R0 = 0.9, 0.1, 0.0  # 初始条件
# sir_solution = solve_sir(beta, gamma, S0, I0, R0, t)
#
# # 解码
# decoded_output = decoder(sir_solution.view(1, -1))
#
