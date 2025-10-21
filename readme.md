# Physics-Informed Neural Networks for the 1D Heat Equation

Learn how neural networks can solve differential equations by encoding physical laws directly into the loss function. This implementation tackles the 1D heat equation from multiple angles: solving it with known parameters, discovering unknown parameters from data, predicting beyond training boundaries, and quantifying prediction uncertainty.

## The Problem

We're working with the classic 1D heat equation:

```
∂u/∂t = α ∂²u/∂x²
```

Where `u(x,t)` is temperature, `α` is thermal diffusivity, with:

- Initial condition: `u(x,0) = sin(πx)`
- Boundary conditions: `u(0,t) = u(1,t) = 0`

## What's Inside

**Forward Problem** — Given the thermal diffusivity, predict temperature distribution over space and time.

**Inverse Problem** — Observe temperature at various points, then work backwards to identify the unknown thermal diffusivity parameter.

**Extrapolation** — Test how well the model predicts beyond the time range it was trained on.

**Uncertainty Quantification** — Use Monte Carlo Dropout to estimate confidence in predictions.

**Validation** — Everything compared against analytical solutions to verify accuracy.

## Requirements

```
torch
numpy
matplotlib
seaborn
scipy
```

## Training Strategy

### Loss Function Components

1. **PDE Residual Loss**: Enforces physical law

   - `L_pde = |∂u/∂t - α∂²u/∂x²|²`

2. **Boundary Condition Loss**: Enforces boundary constraints

   - `L_bc = |u(0,t)|² + |u(1,t)|²`

3. **Initial Condition Loss**: Enforces initial state

   - `L_ic = |u(x,0) - sin(πx)|²`

4. **Data Loss** (inverse problem only): Matches observations
   - `L_data = |u_pred - u_obs|²`

Total loss uses weighted combination with adaptive weights to balance different components.

### Training Details

- **Optimizer**: Adam with learning rate 1e-3 (forward) or 5e-4 (inverse)
- **Epochs**: 20,000
- **Collocation Points**: 2,000 interior + 400 boundary + 200 initial
- **Gradient Clipping**: Applied for stability in inverse problem
- **Parameter Constraints**: α clamped to [1e-6, 1.0]

## Uncertainty Quantification

Monte Carlo Dropout provides epistemic uncertainty estimates:

- 100 forward passes with active dropout during inference
- Mean prediction: best estimate
- Standard deviation: confidence measure
- Uncertainty increases with time as information propagates

Key findings:

- Mean uncertainty at t=0.1: 0.048
- Mean uncertainty at t=0.5: 0.057
- Higher uncertainty near boundaries and at later times

## Usage

```python
# Initialize data generator
domain_bounds = {'x': (0, 1), 't': (0, 0.5)}
data_gen = DataGenerator(domain_bounds, alpha=0.1)

# Generate training data
train_data = data_gen.generate_training_data(
    n_interior=2000,
    n_boundary=400,
    n_initial=200
)

# Forward problem
forward_pinn = ForwardPINN([2, 50, 50, 50, 50, 1], alpha=0.1, device='cuda')
forward_pinn.setup_optimizer(lr=1e-3)
forward_pinn.train(train_data, epochs=20000)

# Make predictions
u_pred = forward_pinn.predict(x_test, t_test)

# Inverse problem with observations
x_obs, t_obs, u_obs = data_gen.generate_observation_data(n_obs=1000, noise_level=0.01)
inverse_pinn = InversePINN([2, 64, 64, 64, 64, 1], alpha_init=0.01, device='cuda')
inverse_pinn.setup_optimizer(lr_model=5e-4, lr_alpha=1e-3)
inverse_pinn.train(train_data, (x_obs, t_obs, u_obs), epochs=20000)

print(f"Estimated alpha: {inverse_pinn.alpha.item():.6f}")
```

## Limitations and Future Work

- Inverse problem achieves ~90% accuracy; could be improved further
- Extrapolation accuracy degrades with time
- Could extend to 2D/3D heat equations

## References

- Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. _Journal of Computational Physics_.
