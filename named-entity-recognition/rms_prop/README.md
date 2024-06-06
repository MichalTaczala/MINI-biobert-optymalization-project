# Custom Optimizer: RMSProp with SGD in PyTorch

## Introduction

This README provides a detailed explanation of the implementation of an optimizer that combines RMSProp with Stochastic Gradient Descent (SGD) using momentum in PyTorch. The provided `step` function is part of a custom optimizer class designed to perform parameter updates during neural network training.

## Implementation Details

The `step` function iterates through all parameter groups and updates each parameter based on the RMSProp and SGD algorithms. The function also includes support for optional momentum and weight decay.

## Code Explanation

Here are the steps:


### Initialize Parameters
```python
if len(state) == 0:
                    state["step"] = 0
                    state["square_avg"] = torch.zeros_like(p.data)
                    if group["momentum"] > 0:
                        state["momentum_buffer"] = torch.zeros_like(p.data)
```


### Extract the Gradient
```python
if p.grad is not None:
    grad = p.grad.data
```


### Update Squared Gradient Moving Average using the formula:
\[ E[g^2]_t = \alpha E[g^2]_{t-1} + (1 - \alpha) g_t^2 \]
where:
- \( E[g^2]_t \) is the moving average of the squared gradients at time step \( t \).
- \( \alpha \) is the decay rate.
- \( E[g^2]_{t-1} \) is the moving average of the squared gradients at the previous time step.
- \( g_t^2 \) is the square of the current gradient.
```python
square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
```

### Compute adjusted learning rate using the following formula:
\[ \text{adjusted\_lr} = \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \]

where:
- \(\eta\) is the learning rate.
- \(E[g^2]_t\) is the moving average of the squared gradients at time step \( t \).
- \(\epsilon\) is a small constant to prevent division by zero.
```python
adjusted_lr = group['lr'] / (square_avg.sqrt().add(group['eps']))
```

### Update Parameters

- With momentum
```python
p.data.addcdiv_(-group['lr'], grad, square_avg.sqrt().add(group['eps']))
```
- Without momentum
```python
buf = state["momentum_buffer"]
buf.mul_(group["momentum"]).addcdiv_(grad, square_avg.sqrt().add(group["eps"]))
p.data.add_(-group['lr'], buf)

```

### Apply momentum
```python
if group["momentum"] > 0:
    buf = state["momentum_buffer"]
    buf.mul_(group["momentum"]).addcdiv_(grad, square_avg.sqrt().add(group["eps"]))
    p.data.add_(-group['lr'], buf)
else:
    p.data.addcdiv_(-group['lr'], grad, square_avg.sqrt().add(group['eps']))
```
The formula for the line `buf.mul_(group["momentum"]).addcdiv_(grad, square_avg.sqrt().add(group["eps"]))` is:

\[ \mathbf{v}_{t+1} = \gamma \mathbf{v}_t + \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \cdot \mathbf{g}_t \]

where:
- \(\mathbf{v}_t\) is the momentum buffer at time step \(t\).
- \(\gamma\) is the momentum coefficient.
- \(\eta\) is the learning rate.
- \(E[g^2]_t\) is the moving average of the squared gradients at time step \(t\).
- \(\epsilon\) is a small constant to prevent division by zero.
- \(\mathbf{g}_t\) is the gradient at time step \(t\).