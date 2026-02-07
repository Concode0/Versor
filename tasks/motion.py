# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want 
# the industry to build upon this "unbending" paradigm.

import torch
import torch.nn as nn
from core.algebra import CliffordAlgebra
from tasks.base import BaseTask
from datasets.har import HARDataset
from models.motion import MotionManifoldNetwork
from torch.utils.data import DataLoader
from core.visualizer import GeneralVisualizer

class MotionAlignmentTask(BaseTask):
    """Motion Alignment Task. Distinguishes between different motion types.

    Aligns complex motion data into linearly separable latent space.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.training_snapshots = []

    def setup_algebra(self):
        """Sets up algebra."""
        return CliffordAlgebra(p=self.cfg.algebra.p, q=self.cfg.algebra.q, device=self.device)

    def setup_model(self):
        """Sets up the Motion Network."""
        input_dim = 561
        num_classes = 6
        
        import os
        if not os.path.exists('./data/HAR/train.csv'):
            input_dim = 45 # AMASS synthetic default
            num_classes = 3
            
        return MotionManifoldNetwork(self.algebra, input_dim=input_dim, latent_dim=self.cfg.algebra.p, num_classes=num_classes)

    def setup_criterion(self):
        """Cross Entropy."""
        return nn.CrossEntropyLoss()

    def get_data(self):
        """Loads HAR or AMASS."""
        try:
            dataset = HARDataset(self.algebra, root='./data/HAR', split='train')
        except FileNotFoundError:
            try:
                from examples.datasets.amass import AMASSDataset
            except ImportError:
                raise FileNotFoundError("HAR data not found and AMASS fallback unavailable.")
            dataset = AMASSDataset(self.algebra, num_samples=self.cfg.dataset.samples)
            
        return DataLoader(dataset, batch_size=self.cfg.training.batch_size, shuffle=True)

    def train_step(self, batch):
        """Learn to separate."""
        data, labels = batch
        data, labels = data.to(self.device), labels.to(self.device)
        
        self.optimizer.zero_grad()
        
        logits, vectors, aligned = self.model(data)
        
        loss = self.criterion(logits, labels)
        
        loss.backward()
        self.optimizer.step()
        
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()
        
        from core.metric import grade_purity
        purity = grade_purity(self.algebra, aligned.squeeze(1), grade=1).mean()
        
        return loss.item(), {
            "Loss": loss.item(), 
            "Acc": acc.item(), 
            "Purity": purity.item()
        }

    def evaluate(self, data=None, noise_std=0.1):
        """Test with noise. Robustness check."""
        self.model.eval()

        if data is None:
            try:
                test_dataset = HARDataset(self.algebra, root='./data/HAR', split='test')
            except FileNotFoundError:
                try:
                    from examples.datasets.amass import AMASSDataset
                except ImportError:
                    raise FileNotFoundError("HAR data not found and AMASS fallback unavailable.")
                test_dataset = AMASSDataset(self.algebra, num_samples=self.cfg.dataset.samples)

            data = DataLoader(test_dataset, batch_size=self.cfg.training.batch_size, shuffle=False)
        elif isinstance(data, (tuple, list)) and len(data) == 2 and isinstance(data[0], torch.Tensor):
            data = [data]

        total_loss = 0.0
        total_acc = 0.0
        total_purity = 0.0
        total_loss_noisy = 0.0
        total_acc_noisy = 0.0
        total_purity_noisy = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in data:
                if isinstance(batch, (tuple, list)):
                    batch_data, labels = batch[0], batch[1]
                else:
                    raise ValueError(f"Unexpected batch format: {type(batch)}")

                batch_data, labels = batch_data.to(self.device), labels.to(self.device)

                # Clean
                logits, _, aligned = self.model(batch_data)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                preds = logits.argmax(dim=1)
                acc = (preds == labels).float().mean()
                total_acc += acc.item()

                from core.metric import grade_purity
                purity = grade_purity(self.algebra, aligned.squeeze(1), grade=1).mean()
                total_purity += purity.item()

                # Noisy
                if noise_std > 0:
                    noise = torch.randn_like(batch_data) * noise_std
                    noisy_data = batch_data + noise

                    logits_noisy, _, aligned_noisy = self.model(noisy_data)

                    loss_noisy = self.criterion(logits_noisy, labels)
                    total_loss_noisy += loss_noisy.item()

                    preds_noisy = logits_noisy.argmax(dim=1)
                    acc_noisy = (preds_noisy == labels).float().mean()
                    total_acc_noisy += acc_noisy.item()

                    purity_noisy = grade_purity(self.algebra, aligned_noisy.squeeze(1), grade=1).mean()
                    total_purity_noisy += purity_noisy.item()

                num_batches += 1

        self.model.train()

        metrics = {
            "Loss": total_loss / num_batches,
            "Acc": total_acc / num_batches,
            "Purity": total_purity / num_batches
        }

        print(f"Evaluation (Clean) - Loss: {metrics['Loss']:.4f}, Acc: {metrics['Acc']:.4f}, Purity: {metrics['Purity']:.4f}")

        if noise_std > 0:
            metrics_noisy = {
                "Loss_Noisy": total_loss_noisy / num_batches,
                "Acc_Noisy": total_acc_noisy / num_batches,
                "Purity_Noisy": total_purity_noisy / num_batches
            }
            print(f"Evaluation (Noisy σ={noise_std}) - Loss: {metrics_noisy['Loss_Noisy']:.4f}, Acc: {metrics_noisy['Acc_Noisy']:.4f}, Purity: {metrics_noisy['Purity_Noisy']:.4f}")

            acc_drop = (metrics['Acc'] - metrics_noisy['Acc_Noisy']) * 100
            print(f"Robustness - Accuracy drop: {acc_drop:.2f}%")

            metrics.update(metrics_noisy)

        return metrics

    def run(self):
        """Captures training history for visualization."""
        from tqdm import tqdm

        print(f">>> Starting Task: motion")
        dataloader = self.get_data()

        snapshot_batch = next(iter(dataloader))
        snapshot_inputs, snapshot_labels = snapshot_batch
        snapshot_inputs = snapshot_inputs.to(self.device)

        is_loader = not isinstance(dataloader, (torch.Tensor, tuple, list))

        self.model.train()
        pbar = tqdm(range(self.epochs))

        snapshot_epochs = set([0, 1, 5, 10, 20, 30, 50, 75, self.epochs-1])

        for epoch in pbar:
            if is_loader:
                total_loss = 0
                for batch in dataloader:
                    loss, logs = self.train_step(batch)
                    total_loss += loss
                avg_loss = total_loss / len(dataloader)
                logs['Loss'] = avg_loss
            else:
                loss, logs = self.train_step(dataloader)

            step_loss = avg_loss if is_loader else loss
            self.scheduler.step(step_loss)

            current_lr = self.optimizer.param_groups[0]['lr']
            logs['LR'] = current_lr

            desc = " | ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
            pbar.set_description(desc)

            if epoch in snapshot_epochs:
                self.model.eval()
                with torch.no_grad():
                    _, vectors, _ = self.model(snapshot_inputs)
                    self.training_snapshots.append({
                        'epoch': epoch,
                        'vectors': vectors.cpu().numpy(),
                        'labels': snapshot_labels.numpy()
                    })
                self.model.train()

        print(">>> Training Complete.")

        self.model.eval()
        with torch.no_grad():
            sample_data = next(iter(dataloader)) if is_loader else dataloader
            self.evaluate(sample_data)
            self.visualize(sample_data)

            if len(self.training_snapshots) > 0:
                self.create_animation()

    def create_animation(self):
        """Generates a training progression animation."""
        try:
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
            import imageio
            import numpy as np

            print(">>> Creating animation of training progression...")

            class_names = ['LAYING', 'SITTING', 'STANDING', 'WALKING', 'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS']
            colors = plt.cm.tab10(np.linspace(0, 1, 6))

            frames = []
            temp_files = []

            final_snapshot = self.training_snapshots[-1]
            pca = PCA(n_components=2)
            pca.fit(final_snapshot['vectors'])

            for snapshot in self.training_snapshots:
                epoch = snapshot['epoch']
                vectors = snapshot['vectors']
                labels = snapshot['labels']

                vecs_2d = pca.transform(vectors)

                fig, ax = plt.subplots(figsize=(10, 8))

                for class_idx, class_name in enumerate(class_names):
                    mask = labels == class_idx
                    if mask.any():
                        ax.scatter(vecs_2d[mask, 0], vecs_2d[mask, 1],
                                 c=[colors[class_idx]],
                                 label=class_name,
                                 alpha=0.6,
                                 s=30,
                                 edgecolors='black',
                                 linewidths=0.5)

                ax.set_xlabel('PC1', fontsize=12)
                ax.set_ylabel('PC2', fontsize=12)
                ax.set_title(f'Motion Latent Space Evolution - Epoch {epoch}', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right', fontsize=10)

                ax.set_xlim(vecs_2d[:, 0].min() - 1, vecs_2d[:, 0].max() + 1)
                ax.set_ylim(vecs_2d[:, 1].min() - 1, vecs_2d[:, 1].max() + 1)

                temp_file = f'/tmp/motion_frame_{epoch}.png'
                plt.tight_layout()
                plt.savefig(temp_file, dpi=100)
                temp_files.append(temp_file)
                frames.append(imageio.imread(temp_file))
                plt.close()

            output_file = 'motion_training_animation.gif'
            imageio.mimsave(output_file, frames, duration=0.8, loop=0)

            import os
            for temp_file in temp_files:
                os.remove(temp_file)

            print(f">>> Saved animation to {output_file}")

        except ImportError as e:
            print(f">>> Could not create animation: {e}")
            print(">>> Install imageio with: pip install imageio")

    def visualize(self, data):
        """Visualizes the latent space and rotor weights."""
        loader = self.get_data()
        batch = next(iter(loader))
        inputs, labels = batch
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            _, vectors, _ = self.model(inputs)
            
        try:
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
            
            vecs = vectors.cpu().numpy()
            lbls = labels.numpy()
            
            if vecs.shape[1] > 2:
                pca = PCA(n_components=2)
                vecs_2d = pca.fit_transform(vecs)
                x_label, y_label = "PC1", "PC2"
                explained_var = pca.explained_variance_ratio_
                title_suffix = f"(PCA: {explained_var[0]:.2f}, {explained_var[1]:.2f})"
            else:
                vecs_2d = vecs
                x_label, y_label = "e1", "e2"
                title_suffix = ""
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(vecs_2d[:, 0], vecs_2d[:, 1], c=lbls, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, label='Activity Class')
            plt.title(f"Motion Latent Space {title_suffix}")
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.grid(True, alpha=0.3)
            plt.savefig("motion_latent_space.png")
            print(">>> Saved visualization to motion_latent_space.png")
            plt.close()
            
            # 2. Bivector Map (Rotor Weights)
            weights = self.model.rotor.bivector_weights.detach().cpu().numpy()
            
            plt.figure(figsize=(10, 4))
            plt.imshow(weights, aspect='auto', cmap='coolwarm', interpolation='nearest')
            plt.colorbar(label='Weight Magnitude')
            plt.title("Learned Bivector Map (Rotor Planes)")
            plt.xlabel("Bivector Basis Index")
            plt.ylabel("Channel")
            plt.savefig("motion_bivector_map.png")
            print(">>> Saved visualization to motion_bivector_map.png")
            plt.close()
            
        except ImportError:
            print("Matplotlib or Sklearn not found, skipping plot.")