import torch
import torch.optim as optim
import tqdm
import abc

class BaseTask(abc.ABC):
    def __init__(self, epochs: int = 100, lr: float = 0.01, device: str = 'cpu'):
        self.epochs = epochs
        self.lr = lr
        self.device = device
        
        self.algebra = self.setup_algebra()
        self.model = self.setup_model().to(self.device)
        self.optimizer = self.setup_optimizer()
        self.criterion = self.setup_criterion()

    @abc.abstractmethod
    def setup_algebra(self):
        """Initialize and return the CliffordAlgebra instance."""
        pass

    @abc.abstractmethod
    def setup_model(self):
        """Initialize and return the Neural Network model."""
        pass

    def setup_optimizer(self):
        """Initialize the optimizer. Defaults to Adam."""
        return optim.Adam(self.model.parameters(), lr=self.lr)

    @abc.abstractmethod
    def setup_criterion(self):
        """Initialize and return the loss function."""
        pass

    @abc.abstractmethod
    def get_data(self):
        """Generate or load data. Returns a tensor or tuple of tensors."""
        pass

    @abc.abstractmethod
    def train_step(self, data):
        """
        Perform a single training step.
        Args:
            data: The data returned by get_data()
        Returns:
            loss (float): The loss value for the step
            metrics (dict): Optional dictionary of metrics to display
        """
        pass

    def run(self):
        """Main execution loop."""
        print(f">>> Starting Task: {self.__class__.__name__}")
        
        # Prepare data
        data = self.get_data()
        if isinstance(data, (tuple, list)):
            data = tuple(d.to(self.device) if isinstance(d, torch.Tensor) else d for d in data)
        else:
            data = data.to(self.device)

        # Training Loop
        self.model.train()
        pbar = tqdm.tqdm(range(self.epochs))
        
        for epoch in pbar:
            metrics = self.train_step(data)
            
            # Handle return formats
            if isinstance(metrics, tuple):
                loss, metric_dict = metrics
            else:
                loss, metric_dict = metrics, {}
                
            desc = f"Loss: {loss:.4f}"
            for k, v in metric_dict.items():
                desc += f" | {k}: {v:.4f}"
            pbar.set_description(desc)
            
        print("\n>>> Training Complete.")
        
        # Evaluation & Visualization
        self.model.eval()
        with torch.no_grad():
            self.evaluate(data)
            self.visualize(data)

    @abc.abstractmethod
    def evaluate(self, data):
        """Run evaluation metrics after training."""
        pass

    @abc.abstractmethod
    def visualize(self, data):
        """Generate and save visualizations."""
        pass
