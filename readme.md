# Physics-Informed Neural Networks (PINNs) for 1D Heat Equation

A working implementation of PINNs to solve the 1D heat equation using neural networks and automatic differentiation.

## The Problem

We're solving the heat equation:
```
∂u/∂t = α ∂²u/∂x²
```

On the domain `[0,1] × [0,1]` with:
- Initial condition: `u(x,0) = sin(πx)`
- Boundary conditions: `u(0,t) = u(1,t) = 0`
- Thermal diffusivity: `α = 0.01`

## How it Works

Instead of solving this with finite differences or finite elements, a neural network learns the solution `u(x,t)` by:

1. **Forward pass**: Takes spatial `x` and temporal `t` as inputs, outputs temperature `u`
2. **Computing derivatives**: Uses PyTorch's autograd to compute `∂u/∂t` and `∂²u/∂x²`
3. **Enforcing physics**: Minimizes three losses simultaneously:
   - PDE loss: How well does `∂u/∂t - α∂²u/∂x²` equal zero?
   - Boundary loss: Does the network respect `u(0,t) = u(1,t) = 0`?
   - Initial loss: Does it match `u(x,0) = sin(πx)`?

The network learns by satisfying the physics constraints, not by memorizing training data.

## Code Structure

- **PINN class**: Simple 2-layer MLP with Xavier initialization and Tanh activation
- **pde_residual()**: Computes second derivatives using autograd
- **compute_losses()**: Calculates PDE, boundary, and initial condition losses
- **train_pinn()**: Adam optimizer with learning rate scheduling and early stopping
- **Evaluation**: Compares predictions against the analytical solution at different times

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
jupyter notebook pinn_heat_equation.ipynb
```

The notebook trains the network and visualizes:
- Loss curves during training
- Predictions vs analytical solution at different time steps
- Full spatiotemporal heatmaps
- PDE residual statistics
