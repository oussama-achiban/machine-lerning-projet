"""
Neural Network Module using PyTorch
Implements: MLP (Multi-Layer Perceptron) with custom Dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class CustomDataset(Dataset):
    """Custom PyTorch Dataset"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    """Multi-Layer Perceptron"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int,
                dropout_rate: float = 0.2):
        super(MLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class NeuralNetworkTrainer:
    """Trains and evaluates neural networks"""
    
    def __init__(self, input_size: int, output_size: int, hidden_sizes: List[int] = None,
                learning_rate: float = 0.001, device: str = None):
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]
        
        print(f"Using device: {self.device}")
        print(f"Network architecture: {input_size} -> {hidden_sizes} -> {output_size}")
        
        self.model = MLP(input_size, hidden_sizes, output_size).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            # Forward pass
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == y_batch).sum().item()
            total_samples += y_batch.size(0)
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                total_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == y_batch).sum().item()
                total_samples += y_batch.size(0)
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
           X_val: np.ndarray = None, y_val: np.ndarray = None,
           epochs: int = 100, batch_size: int = 32, verbose: int = 10):
        """Train the neural network"""
        
        print("\n" + "="*60)
        print("TRAINING NEURAL NETWORK")
        print("="*60)
        
        # Create datasets
        train_dataset = CustomDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None and y_val is not None:
            val_dataset = CustomDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None
        
        # Training loop
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            if val_loader is not None:
                val_loss, val_acc = self.validate(val_loader)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)
                
                if (epoch + 1) % verbose == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] | "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                if (epoch + 1) % verbose == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] | "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        print("="*60)
        print("TRAINING COMPLETE")
        print("="*60)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        self.model.eval()
        X_tensor = torch.FloatTensor(np.asarray(X)).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predictions = torch.max(outputs, 1)
        
        return predictions.cpu().numpy()
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluate on test set"""
        y_pred = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        # Use macro-average to support multiclass targets
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        print("\n" + "="*60)
        print("NEURAL NETWORK EVALUATION (Test Set)")
        print("="*60)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print("="*60)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def plot_training_history(self, figsize: Tuple = (12, 5)):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Loss
        axes[0].plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            axes[0].plot(self.val_losses, label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Accuracy
        axes[1].plot(self.train_accuracies, label='Train Accuracy')
        if self.val_accuracies:
            axes[1].plot(self.val_accuracies, label='Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training Accuracy')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        axes[1].set_ylim([0, 1])
        
        plt.tight_layout()
        return fig
    
    def save_model(self, filepath: str):
        """Save model to file"""
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file"""
        # Ensure model is loaded to the trainer's device
        state = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(state)
        print(f"Model loaded from {filepath}")


def get_neural_network_trainer(input_size: int, output_size: int,
                               hidden_sizes: List[int] = None) -> NeuralNetworkTrainer:
    """Factory function to create trainer"""
    return NeuralNetworkTrainer(input_size, output_size, hidden_sizes)
