import torch
import torch.nn as nn
import torch.optim as optim
import math
import tqdm
import matplotlib.pyplot as plt
from core.algebra import CliffordAlgebra
from layers.rotor import RotorLayer
from functional.loss import GeometricMSELoss

class BoostLearner(nn.Module):
    def __init__(self, algebra):
        super().__init__()
        self.algebra = algebra
        # Learn a single rotor to undo the boost
        self.rotor = RotorLayer(algebra, channels=1)

    def forward(self, x):
        return self.rotor(x)

def generate_hyperbola_data(num_samples=1000):
    # x^2 - t^2 = 1 (Unit hyperbola)
    # x = cosh(eta), t = sinh(eta)
    eta = torch.linspace(-2, 2, num_samples)
    x = torch.cosh(eta)
    t = torch.sinh(eta)
    
    # In R1,1: e1 is Space, e2 is Time
    return torch.stack([x, t], dim=1)

def apply_random_boost(points, algebra, boost_factor=1.0):
    # Apply a fixed boost to the data
    # R = exp(-phi/2 * e12)
    # e12 in R1,1 squares to +1?
    # e1*e1 = 1, e2*e2 = -1
    # e1e2 * e1e2 = -e1e1 e2e2 = -(1)(-1) = 1.
    # So Taylor series gives cosh - sinh.
    
    # Create the boost rotor manually
    # phi = boost_factor
    phi = boost_factor
    
    # Bivector e1e2 is index 3 (1 | 2)
    B = torch.zeros(1, algebra.dim)
    B[0, 3] = 1.0 
    
    # R = exp(-phi/2 * B)
    R = algebra.exp(-0.5 * phi * B)
    R_rev = algebra.reverse(R)
    
    # Apply to points
    # Convert points to MV: x*e1 + t*e2
    batch_size = points.shape[0]
    mv = torch.zeros(batch_size, 1, algebra.dim)
    mv[:, 0, 1] = points[:, 0]
    mv[:, 0, 2] = points[:, 1]
    
    # RxR~
    # Broadcast R
    R_exp = R.unsqueeze(0) # [1, 1, D]
    R_rev_exp = R_rev.unsqueeze(0)
    
    boosted = algebra.geometric_product(R_exp, mv)
    boosted = algebra.geometric_product(boosted, R_rev_exp)
    
    return mv, boosted, phi

def run_hyperbolic_task(epochs=500, lr=0.05):
    # R1,1
    device = 'cpu'
    algebra = CliffordAlgebra(p=1, q=1, device=device)
    
    print(">>> [Task] Hyperbolic Geometry (Learning Lorentz Boost)...")
    
    # 1. Generate Data
    original_mv, boosted_mv, true_phi = apply_random_boost(
        generate_hyperbola_data(), algebra, boost_factor=1.5
    )
    
    print(f"True Boost Parameter (Phi): {true_phi}")
    
    # 2. Model
    # We want to learn R_inv such that R_inv * Boosted * R_inv~ = Original
    model = BoostLearner(algebra).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = GeometricMSELoss(algebra)
    
    pbar = tqdm.tqdm(range(epochs))
    
    for epoch in pbar:
        optimizer.zero_grad()
        
        recovered = model(boosted_mv)
        
        # Loss: Distance to original points
        loss = criterion(recovered, original_mv)
        
        loss.backward()
        optimizer.step()
        
        pbar.set_description(f"Loss: {loss.item():.6f}")
        
    print("\n>>> Evaluation...")
    
    # Check learned parameter
    # The model learned a bivector B_learn.
    # The angle should be approx -1.5 (to undo +1.5)
    learned_weights = model.rotor.bivector_weights.detach()
    # In R1,1, there is only 1 bivector (e12).
    # RotorLayer constructs B = weights * Basis.
    # The weight is effectively the angle phi (or related).
    
    print(f"Learned Rotor Weights (Bivector Coeffs): {learned_weights.flatten().numpy()}")
    
    # Visualization
    img_counter = [1]
    def save_fig_and_close():
        filename = f"hyperbolic_viz_{img_counter[0]}.png"
        plt.savefig(filename)
        print(f"Saved {filename}")
        plt.close()
        img_counter[0] += 1
    plt.show = save_fig_and_close

    # Plot
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(131)
    plot_2d(ax1, original_mv, "Original (Rest Frame)")
    
    ax2 = fig.add_subplot(132)
    plot_2d(ax2, boosted_mv, f"Boosted (Phi={true_phi})")
    
    ax3 = fig.add_subplot(133)
    with torch.no_grad():
        recovered = model(boosted_mv)
    plot_2d(ax3, recovered, "Recovered")
    
    plt.tight_layout()
    plt.show()

def plot_2d(ax, mv_data, title):
    x = mv_data[:, 0, 1].numpy()
    t = mv_data[:, 0, 2].numpy()
    ax.scatter(x, t, s=2, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel('x (Space)')
    ax.set_ylabel('t (Time)')
    ax.grid(True)
    ax.set_aspect('equal')
