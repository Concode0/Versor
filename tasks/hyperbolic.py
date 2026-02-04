import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from core.algebra import CliffordAlgebra
from layers.rotor import RotorLayer
from functional.loss import GeometricMSELoss
from tasks.base import BaseTask
from core.visualizer import GeneralVisualizer

class BoostLearner(nn.Module):
    def __init__(self, algebra):
        super().__init__()
        self.algebra = algebra
        # Learn a single rotor to undo the boost
        self.rotor = RotorLayer(algebra, channels=1)

    def forward(self, x):
        return self.rotor(x)

class HyperbolicTask(BaseTask):
    def setup_algebra(self):
        # R1,1: p=1, q=1
        return CliffordAlgebra(p=1, q=1, device=self.device)

    def setup_model(self):
        return BoostLearner(self.algebra)

    def setup_criterion(self):
        return GeometricMSELoss(self.algebra)

    def get_data(self):
        num_samples = 1000
        # x^2 - t^2 = 1 (Unit hyperbola)
        eta = torch.linspace(-2, 2, num_samples)
        x = torch.cosh(eta)
        t = torch.sinh(eta)
        
        # In R1,1: e1 is Space, e2 is Time
        # points: [N, 2]
        points = torch.stack([x, t], dim=1)
        
        # Apply random boost
        boost_factor = 1.5
        true_phi = boost_factor
        
        # Create the boost rotor manually
        B = torch.zeros(1, self.algebra.dim)
        B[0, 3] = 1.0 # e1e2
        
        R = self.algebra.exp(-0.5 * true_phi * B)
        R_rev = self.algebra.reverse(R)
        
        # Original MV
        original_mv = torch.zeros(num_samples, 1, self.algebra.dim)
        original_mv[:, 0, 1] = points[:, 0]
        original_mv[:, 0, 2] = points[:, 1]
        
        # Boosted MV
        R_exp = R.unsqueeze(0)
        R_rev_exp = R_rev.unsqueeze(0)
        boosted_mv = self.algebra.geometric_product(R_exp, original_mv)
        boosted_mv = self.algebra.geometric_product(boosted_mv, R_rev_exp)
        
        return original_mv, boosted_mv, true_phi

    def train_step(self, data):
        original_mv, boosted_mv, true_phi = data
        
        self.optimizer.zero_grad()
        
        recovered = self.model(boosted_mv)
        loss = self.criterion(recovered, original_mv)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), {}

    def evaluate(self, data):
        original_mv, boosted_mv, true_phi = data
        
        learned_weights = self.model.rotor.bivector_weights.detach()
        print(f"True Phi: {true_phi}")
        print(f"Learned Rotor Weights: {learned_weights.flatten().cpu().numpy()}")
        
        recovered = self.model(boosted_mv)
        final_loss = self.criterion(recovered, original_mv)
        print(f"Final Reconstruction Loss: {final_loss.item():.6f}")

    def visualize(self, data):
        original_mv, boosted_mv, true_phi = data
        recovered = self.model(boosted_mv)
        
        # Custom 2D Plot for R1,1
        fig = plt.figure(figsize=(12, 4))
        ax1 = fig.add_subplot(131)
        self._plot_2d(ax1, original_mv, "Original (Rest Frame)")
        
        ax2 = fig.add_subplot(132)
        self._plot_2d(ax2, boosted_mv, f"Boosted (Phi={true_phi})")
        
        ax3 = fig.add_subplot(133)
        self._plot_2d(ax3, recovered, "Recovered")
        
        plt.tight_layout()
        filename = "hyperbolic_viz.png"
        plt.savefig(filename)
        print(f"Saved {filename}")
        plt.close()
        
        # Also use GeneralVisualizer for heatmaps
        viz = GeneralVisualizer(self.algebra)
        viz.plot_grade_heatmap(recovered, title="Recovered Grade Energy")
        viz.save("hyperbolic_grades.png")

    def _plot_2d(self, ax, mv_data, title):
        x = mv_data[:, 0, 1].cpu().numpy()
        t = mv_data[:, 0, 2].cpu().numpy()
        ax.scatter(x, t, s=2, alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel('x (Space)')
        ax.set_ylabel('t (Time)')
        ax.grid(True)
        ax.set_aspect('equal')

def run_hyperbolic_task(epochs=500, lr=0.05):
    task = HyperbolicTask(epochs=epochs, lr=lr, device='cpu')
    task.run()
