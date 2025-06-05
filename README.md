# micrograd

This is a hands-on educational implementation to understand backpropagation (reverse-mode autodiff) from scratch over a dynamically built DAG of operations.

**Inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd)**

The core idea is to start with individual scalar values and build up neural networks by composing them through basic mathematical operations (`+`, `-`, `*`, `/`, `**`). Each operation tracks its inputs and knows how to compute gradients, enabling automatic differentiation through the chain rule. This approach provides deep insight into how backpropagation works under the hood in modern deep learning frameworks.

The entire neural network training process - forward pass, loss computation, backward pass, and parameter updates - is implemented from first principles using only basic math operations on scalars.

## Installation

Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib ipykernel
```

## Example usage

Basic operations with automatic differentiation:

```python
from micrograd.engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
e = c - d
f = e**2
g = f / 2.0

print(f'{g.data:.4f}')  # prints the outcome of this forward pass
g.backward()
print(f'{a.grad:.4f}')  # prints the numerical value of dg/da
print(f'{b.grad:.4f}')  # prints the numerical value of dg/db
```

## Project Structure

```
micrograd/
├── __init__.py         # Package initialization
├── engine.py           # Core Value class with autograd
├── function.py         # MSE loss and ReLU functions
└── nn.py              # Neural network components

demo.ipynb             # Student performance prediction demo
student_habits_performance.csv  # Dataset with study habits and exam scores
test.py                # Test file
LICENSE                # MIT license
pyproject.toml         # Project configuration
```

## Features

### Core Engine (`engine.py`)
- **Value class**: Scalar value wrapper with automatic differentiation
- **Operations**: `+`, `-`, `*`, `/`, `**` with proper gradient computation
- **Backpropagation**: Automatic gradient computation via topological sort
- **Type safety**: Full type annotations for better development experience

### Neural Networks (`nn.py`)
- **Module**: Base class with `zero_grad()` and `parameters()` methods
- **Neuron**: Single neuron with random weight initialization
- **Layer**: Collection of neurons forming a layer
- **MLP**: Multi-layer perceptron with configurable architecture
- **MSELoss**: Mean squared error loss for regression tasks
- **ReLU**: ReLU activation function

### Functions (`function.py`)
- **MSE Loss**: `(prediction - target)²` averaged over batch
- **ReLU**: `max(0, x)` with proper gradient handling

## Demo: Student Performance Prediction

The `demo.ipynb` notebook demonstrates a complete machine learning pipeline:

### Dataset
- **Source**: `student_habits_performance.csv`
- **Features**: Study habits, parental education, diet quality, etc.
- **Target**: Exam scores (regression task)
- **Preprocessing**: StandardScaler, categorical encoding, train/test split

### Model Architecture
- **Input**: 15 features after preprocessing
- **Architecture**: Linear layer (15 → 1)
- **Loss**: Mean Squared Error
- **Optimizer**: SGD with learning rate decay (0.05 → ~0.001)

### Training Process
```python
# Data preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model training
model = nn.MLP(15, [1])  # Single linear layer
for step in range(200):
    # Forward pass
    y_pred = [model(x) for x in X_train]
    loss = loss_fn(y_pred, y_train)
    
    # Gradient computation and parameter update
    model.zero_grad()
    loss.backward()
    
    # Learning rate decay
    lr = 0.05 - 0.049 * step / 200
    for p in model.parameters():
        p.data = p.data - lr * p.grad
```

### Results
- Evaluates model performance using R² score on test set
- Demonstrates the effectiveness of the micrograd implementation

## Running Tests

To run tests:
```bash
python test.py
```