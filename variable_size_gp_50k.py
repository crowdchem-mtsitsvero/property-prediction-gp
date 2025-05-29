"""
Efficient GP Model for 50,000 Tasks with Variable Training Sample Sizes (1-10 samples per task)
using Deep Kernel Learning with Shared Neural Network

This implementation uses grouped batch processing where tasks are grouped by their number 
of training samples, with a shared 2-layer neural network that all GPs use as a feature extractor.
"""

import math
import torch
import torch.nn as nn
import gpytorch
import numpy as np
from matplotlib import pyplot as plt
import time
import gc
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

class SharedFeatureExtractor(nn.Module):
    """
    Shared 2-layer neural network for deep kernel learning.
    All GPs will use this same feature extractor.
    """
    
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 2-layer neural network
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()  # Optional: can be removed for more flexibility
        )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        """
        Transform input through the neural network.
        
        Args:
            x: Input tensor of shape [..., input_dim]
            
        Returns:
            Transformed features of shape [..., output_dim]
        """
        return self.network(x)

class DeepKernelVariableSizeBatchGPModel(gpytorch.models.ExactGP):
    """GP model with shared deep kernel for batch processing tasks with the same number of training points."""
    
    def __init__(self, train_x, train_y, likelihood, num_tasks_in_batch, num_train_points, shared_feature_extractor):
        super().__init__(train_x, train_y, likelihood)
        self.num_tasks_in_batch = num_tasks_in_batch
        self.num_train_points = num_train_points
        self.shared_feature_extractor = shared_feature_extractor
        
        # Use batch shape for efficient computation across tasks in current batch
        batch_shape = torch.Size([num_tasks_in_batch])
        
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        
        # Deep kernel: RBF kernel operating on neural network features
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=shared_feature_extractor.output_dim,  # Use NN output dimension
                batch_shape=batch_shape
            ),
            batch_shape=batch_shape
        )

    def forward(self, x):
        # Transform input through shared neural network
        # x shape: [num_train_points, input_dim] or [batch_size, num_train_points, input_dim]
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # Add feature dimension if needed
        
        # Pass through shared feature extractor
        transformed_x = self.shared_feature_extractor(x)
        
        # Compute mean and covariance on transformed features
        mean_x = self.mean_module(transformed_x)
        covar_x = self.covar_module(transformed_x)
        
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )

class DeepKernelVariableSizeTaskPredictor:
    """
    TaskPredictor that handles tasks with variable numbers of training samples using deep kernels.
    
    All GPs share the same neural network feature extractor.
    """
    
    def __init__(self, trained_models_by_size: Dict[int, List], 
                 trained_likelihoods_by_size: Dict[int, List],
                 shared_feature_extractor: SharedFeatureExtractor,
                 batch_sizes_by_size: Dict[int, int],
                 task_assignments: Dict[int, Tuple[int, int, int]],
                 device):
        """
        Args:
            shared_feature_extractor: The shared neural network feature extractor
        """
        self.trained_models_by_size = trained_models_by_size
        self.trained_likelihoods_by_size = trained_likelihoods_by_size
        self.shared_feature_extractor = shared_feature_extractor
        self.batch_sizes_by_size = batch_sizes_by_size
        self.task_assignments = task_assignments
        self.device = device
        self.total_tasks = len(task_assignments)
    
    def predict_task(self, task_idx: int, test_x: torch.Tensor, 
                     return_std: bool = False, return_full_dist: bool = False) -> Dict:
        """Make predictions for a specific task by its global index."""
        
        if task_idx not in self.task_assignments:
            raise ValueError(f"Task index {task_idx} not found")
        
        num_points, batch_idx, pos_in_batch = self.task_assignments[task_idx]
        
        # Get the appropriate model and likelihood
        models = self.trained_models_by_size[num_points]
        likelihoods = self.trained_likelihoods_by_size[num_points]
        
        if batch_idx >= len(models):
            raise ValueError(f"Batch {batch_idx} not found for {num_points}-point tasks")
        
        model = models[batch_idx].to(self.device)
        likelihood = likelihoods[batch_idx].to(self.device)
        self.shared_feature_extractor.to(self.device)
        
        try:
            model.eval()
            likelihood.eval()
            self.shared_feature_extractor.eval()
            
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                predictions = likelihood(model(test_x))
                
                # Extract predictions for the specific task
                mean = predictions.mean[:, pos_in_batch]
                
                result = {'mean': mean, 'num_train_points': num_points}
                
                if return_std or return_full_dist:
                    variance = predictions.variance[:, pos_in_batch]
                    std = torch.sqrt(variance)
                    result['std'] = std
                    result['variance'] = variance
                
                if return_full_dist:
                    lower, upper = predictions.confidence_region()
                    result['lower_ci'] = lower[:, pos_in_batch]
                    result['upper_ci'] = upper[:, pos_in_batch]
                
                return result
                
        finally:
            model.cpu()
            likelihood.cpu()
            # Keep feature extractor on device for potential reuse
    
    def predict_multiple_tasks(self, task_indices: List[int], test_x: torch.Tensor, 
                             return_std: bool = False) -> Dict:
        """Make predictions for multiple tasks efficiently."""
        
        self.shared_feature_extractor.to(self.device)
        self.shared_feature_extractor.eval()
        
        # Group tasks by (num_points, batch_idx) for efficiency
        task_groups = defaultdict(list)
        for task_idx in task_indices:
            if task_idx in self.task_assignments:
                num_points, batch_idx, pos_in_batch = self.task_assignments[task_idx]
                task_groups[(num_points, batch_idx)].append((task_idx, pos_in_batch))
        
        # Process each group
        predictions_dict = {}
        for (num_points, batch_idx), tasks_in_group in task_groups.items():
            models = self.trained_models_by_size[num_points]
            likelihoods = self.trained_likelihoods_by_size[num_points]
            
            model = models[batch_idx].to(self.device)
            likelihood = likelihoods[batch_idx].to(self.device)
            
            try:
                model.eval()
                likelihood.eval()
                
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    predictions = likelihood(model(test_x))
                    
                    for task_idx, pos_in_batch in tasks_in_group:
                        mean = predictions.mean[:, pos_in_batch]
                        pred_result = {'mean': mean, 'num_train_points': num_points}
                        
                        if return_std:
                            variance = predictions.variance[:, pos_in_batch]
                            std = torch.sqrt(variance)
                            pred_result['std'] = std
                        
                        predictions_dict[task_idx] = pred_result
            
            finally:
                model.cpu()
                likelihood.cpu()
        
        # Return results in the same order as requested
        means = []
        stds = [] if return_std else None
        num_train_points = []
        
        for task_idx in task_indices:
            if task_idx in predictions_dict:
                pred = predictions_dict[task_idx]
                means.append(pred['mean'])
                num_train_points.append(pred['num_train_points'])
                if return_std:
                    stds.append(pred['std'])
            else:
                # Handle missing tasks
                means.append(None)
                num_train_points.append(None)
                if return_std:
                    stds.append(None)
        
        result = {'means': means, 'num_train_points': num_train_points}
        if return_std:
            result['stds'] = stds
        
        return result
    
    def get_task_info(self, task_idx: int) -> Dict:
        """Get information about a specific task."""
        if task_idx not in self.task_assignments:
            return None
        
        num_points, batch_idx, pos_in_batch = self.task_assignments[task_idx]
        return {
            'task_idx': task_idx,
            'num_train_points': num_points,
            'batch_idx': batch_idx,
            'position_in_batch': pos_in_batch
        }
    
    def get_tasks_by_num_points(self, num_points: int) -> List[int]:
        """Get all task indices that have a specific number of training points."""
        return [task_idx for task_idx, (n, _, _) in self.task_assignments.items() if n == num_points]

def generate_variable_size_tasks(total_num_tasks: int, min_points: int = 1, max_points: int = 10) -> Tuple:
    """
    Generate task parameters and assign number of training points to each task.
    
    Returns:
        tuple: (amplitudes, frequencies, phases, noise_levels, num_train_points_per_task)
    """
    print(f"Generating parameters for {total_num_tasks} tasks with {min_points}-{max_points} training points each...")
    
    # Generate random parameters for each task
    amplitudes = torch.rand(total_num_tasks) * 2 + 0.5
    frequencies = torch.rand(total_num_tasks) * 4 + 1
    phases = torch.rand(total_num_tasks) * 2 * math.pi
    noise_levels = torch.rand(total_num_tasks) * 0.1 + 0.05
    
    # Assign number of training points to each task (1 to 10)
    num_train_points_per_task = torch.randint(min_points, max_points + 1, (total_num_tasks,))
    
    # Print distribution
    point_counts = torch.bincount(num_train_points_per_task, minlength=max_points+1)
    print(f"Distribution of training points:")
    for i in range(min_points, max_points + 1):
        print(f"  {i} points: {point_counts[i]:,} tasks ({point_counts[i]/total_num_tasks*100:.1f}%)")
    
    return amplitudes, frequencies, phases, noise_levels, num_train_points_per_task

def organize_tasks_by_size(total_num_tasks: int, num_train_points_per_task: torch.Tensor,
                          target_memory_gb: float = 6.0) -> Tuple[Dict, Dict]:
    """
    Organize tasks into groups by number of training points and determine batch sizes.
    
    Returns:
        tuple: (task_groups, batch_sizes_by_size)
    """
    min_points = num_train_points_per_task.min().item()
    max_points = num_train_points_per_task.max().item()
    
    # Group tasks by number of training points
    task_groups = defaultdict(list)
    for task_idx in range(total_num_tasks):
        num_points = num_train_points_per_task[task_idx].item()
        task_groups[num_points].append(task_idx)
    
    # Determine batch sizes for each group
    batch_sizes_by_size = {}
    for num_points in range(min_points, max_points + 1):
        if num_points in task_groups:
            num_tasks_in_group = len(task_groups[num_points])
            
            # Estimate memory usage per task (roughly proportional to num_points^2 for GP)
            memory_factor = num_points ** 1.5  # Conservative estimate
            base_batch_size = max(50, min(1000, int(target_memory_gb * 1000 / memory_factor)))
            
            # Ensure batch size doesn't exceed number of tasks in group
            batch_size = min(base_batch_size, num_tasks_in_group)
            batch_sizes_by_size[num_points] = batch_size
            
            print(f"  {num_points} points: {num_tasks_in_group:,} tasks, batch size: {batch_size}")
    
    return dict(task_groups), batch_sizes_by_size

def generate_batch_data_variable_size(task_indices: List[int], num_train_points: int,
                                    amplitudes: torch.Tensor, frequencies: torch.Tensor,
                                    phases: torch.Tensor, noise_levels: torch.Tensor) -> Tuple:
    """Generate training data for a batch of tasks with the same number of training points."""
    
    num_tasks_in_batch = len(task_indices)
    
    # Create training inputs for this number of points
    train_x = torch.linspace(0, 1, num_train_points, device=device)
    
    # Get parameters for this batch
    task_indices_tensor = torch.tensor(task_indices)
    batch_amplitudes = amplitudes[task_indices_tensor].to(device)
    batch_frequencies = frequencies[task_indices_tensor].to(device)
    batch_phases = phases[task_indices_tensor].to(device)
    batch_noise_levels = noise_levels[task_indices_tensor].to(device)
    
    # Generate training targets for batch
    x_expanded = train_x.unsqueeze(0).expand(num_tasks_in_batch, -1)
    
    train_y = (batch_amplitudes.unsqueeze(1) * 
               torch.sin(batch_frequencies.unsqueeze(1) * x_expanded * 2 * math.pi + batch_phases.unsqueeze(1)) +
               torch.randn(num_tasks_in_batch, num_train_points, device=device) * batch_noise_levels.unsqueeze(1))
    
    # Transpose to match expected format: [num_points, num_tasks_in_batch]
    train_y = train_y.t()
    
    return train_x, train_y

def train_deep_kernel_batch(model, likelihood, shared_feature_extractor, train_x, train_y, 
                           training_iterations=20, initial_lr=0.05, feature_lr=0.01):
    """
    Train a batch of deep kernel GPs with shared feature extractor.
    
    Args:
        model: The GP model for this batch
        likelihood: The likelihood for this batch  
        shared_feature_extractor: The shared neural network
        train_x, train_y: Training data
        training_iterations: Number of training iterations
        initial_lr: Learning rate for GP parameters
        feature_lr: Learning rate for neural network parameters
    """
    
    model.train()
    likelihood.train()
    shared_feature_extractor.train()
    
    # Properly separate parameters to avoid duplicates
    # GP parameters: only mean_module and covar_module (excluding shared feature extractor)
    gp_parameters = (
        list(model.mean_module.parameters()) + 
        list(model.covar_module.parameters()) +
        list(likelihood.parameters())
    )
    
    # Neural network parameters: only the shared feature extractor
    nn_parameters = list(shared_feature_extractor.parameters())
    
    # Use different learning rates
    optimizer = torch.optim.Adam([
        {'params': gp_parameters, 'lr': initial_lr},
        {'params': nn_parameters, 'lr': feature_lr}
    ])
    
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    losses = []
    
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(gp_parameters, max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(nn_parameters, max_norm=0.5)  # Smaller for NN
        
        optimizer.step()
        
        losses.append(loss.item())
        
        # Simple early stopping
        if i > 8 and len(losses) > 4:
            if abs(losses[-1] - losses[-4]) < 0.001:
                break
    
    return losses

def main():
    print("=== Deep Kernel Variable Size GP Training for 50,000 Tasks (1-10 training points) ===\n")
    
    # Configuration
    total_num_tasks = 50000
    min_train_points = 1
    max_train_points = 10
    target_memory_gb = 6.0
    
    # Neural network configuration
    nn_hidden_dim = 64
    nn_output_dim = 32
    
    training_iterations_by_size = {
        1: 15,   # Tasks with 1 point need fewer iterations
        2: 20,   # Tasks with 2 points
        3: 25,   # Tasks with 3 points
        4: 30,   # And so on...
        5: 30,
        6: 30,
        7: 35,
        8: 35,
        9: 35,
        10: 35
    }
    
    print(f"Configuration:")
    print(f"  Total tasks: {total_num_tasks:,}")
    print(f"  Training points per task: {min_train_points}-{max_train_points}")
    print(f"  Target memory usage: {target_memory_gb} GB")
    print(f"  Neural network: 1 -> {nn_hidden_dim} -> {nn_output_dim}")
    
    # Create shared feature extractor
    print(f"\nInitializing shared feature extractor...")
    shared_feature_extractor = SharedFeatureExtractor(
        input_dim=1, 
        hidden_dim=nn_hidden_dim, 
        output_dim=nn_output_dim
    ).to(device)
    
    print(f"  Feature extractor parameters: {sum(p.numel() for p in shared_feature_extractor.parameters()):,}")
    
    # Generate task parameters and training point assignments
    amplitudes, frequencies, phases, noise_levels, num_train_points_per_task = \
        generate_variable_size_tasks(total_num_tasks, min_train_points, max_train_points)
    
    # Organize tasks by number of training points
    print(f"\nOrganizing tasks by number of training points...")
    task_groups, batch_sizes_by_size = organize_tasks_by_size(
        total_num_tasks, num_train_points_per_task, target_memory_gb
    )
    
    # Storage for trained models
    trained_models_by_size = defaultdict(list)
    trained_likelihoods_by_size = defaultdict(list)
    task_assignments = {}  # global_task_idx -> (num_points, batch_idx, pos_in_batch)
    training_stats = []
    
    print(f"\nStarting deep kernel grouped batch training...")
    total_start_time = time.time()
    
    # Train each group (by number of training points)
    for group_idx, num_points in enumerate(sorted(task_groups.keys())):
        group_tasks = task_groups[num_points]
        batch_size = batch_sizes_by_size[num_points]
        num_batches = (len(group_tasks) + batch_size - 1) // batch_size
        
        print(f"\n=== Training {num_points}-point tasks (Group {group_idx+1}/10) ===")
        print(f"  Tasks in group: {len(group_tasks):,}")
        print(f"  Batch size: {batch_size}")
        print(f"  Number of batches: {num_batches}")
        
        group_start_time = time.time()
        
        # Train batches for this group
        for batch_idx in range(num_batches):
            batch_start_time = time.time()
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(group_tasks))
            batch_task_indices = group_tasks[start_idx:end_idx]
            current_batch_size = len(batch_task_indices)
            
            print(f"  Batch {batch_idx + 1}/{num_batches}: {current_batch_size} tasks")
            
            # Generate training data for this batch
            train_x, train_y = generate_batch_data_variable_size(
                batch_task_indices, num_points, amplitudes, frequencies, phases, noise_levels
            )
            
            # Initialize model and likelihood
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=current_batch_size
            ).to(device)
            
            model = DeepKernelVariableSizeBatchGPModel(
                train_x, train_y, likelihood, current_batch_size, num_points, shared_feature_extractor
            ).to(device)
            
            # Train the model with shared feature extractor
            training_iterations = training_iterations_by_size.get(num_points, 30)
            
            # Adjust learning rates based on group progress
            feature_lr = 0.01 if group_idx < 3 else 0.005  # Lower LR for NN in later groups
            gp_lr = 0.05
            
            batch_losses = train_deep_kernel_batch(
                model, likelihood, shared_feature_extractor, train_x, train_y, 
                training_iterations, gp_lr, feature_lr
            )
            
            # Record task assignments
            for pos_in_batch, global_task_idx in enumerate(batch_task_indices):
                task_assignments[global_task_idx] = (num_points, batch_idx, pos_in_batch)
            
            # Store trained model and likelihood (but not the shared feature extractor)
            model.eval()
            likelihood.eval()
            trained_models_by_size[num_points].append(model.cpu())
            trained_likelihoods_by_size[num_points].append(likelihood.cpu())
            
            batch_time = time.time() - batch_start_time
            final_loss = batch_losses[-1] if batch_losses else float('inf')
            
            training_stats.append({
                'num_points': num_points,
                'batch_idx': batch_idx,
                'num_tasks': current_batch_size,
                'final_loss': final_loss,
                'training_time': batch_time,
                'num_iterations': len(batch_losses)
            })
            
            print(f"    Final loss: {final_loss:.3f}, Time: {batch_time:.2f}s")
            
            # Clean up GPU memory (keep shared feature extractor)
            del model, likelihood, train_x, train_y
            torch.cuda.empty_cache()
            gc.collect()
        
        group_time = time.time() - group_start_time
        print(f"  Group completed in {group_time/60:.1f} minutes")
        
        # Print feature extractor stats
        feature_norms = []
        for param in shared_feature_extractor.parameters():
            feature_norms.append(param.data.norm().item())
        avg_norm = np.mean(feature_norms)
        print(f"  Shared NN average parameter norm: {avg_norm:.4f}")
    
    total_training_time = time.time() - total_start_time
    print(f"\nAll groups trained successfully!")
    print(f"Total training time: {total_training_time/60:.1f} minutes")
    
    # Move shared feature extractor to CPU to save memory
    shared_feature_extractor.cpu()
    
    # Create the DeepKernelVariableSizeTaskPredictor
    task_predictor = DeepKernelVariableSizeTaskPredictor(
        dict(trained_models_by_size),
        dict(trained_likelihoods_by_size),
        shared_feature_extractor,
        batch_sizes_by_size,
        task_assignments,
        device
    )
    
    print(f"\nCreated DeepKernelVariableSizeTaskPredictor for {task_predictor.total_tasks} tasks")
    
    # Demonstrate predictions for different task sizes
    print(f"\n=== Deep Kernel Variable Size Prediction Examples ===")
    
    # Example predictions for tasks with different numbers of training points
    for num_points in [1, 3, 5, 10]:
        tasks_with_n_points = task_predictor.get_tasks_by_num_points(num_points)
        if tasks_with_n_points:
            example_task = tasks_with_n_points[0]
            test_x = torch.linspace(0, 1, 50, device=device)
            
            try:
                result = task_predictor.predict_task(
                    example_task, test_x, return_std=True
                )
                print(f"  Task {example_task} ({num_points} training points):")
                print(f"    Prediction shape: {result['mean'].shape}")
                print(f"    Mean at x=0.5: {result['mean'][25].item():.4f}")
                print(f"    Std at x=0.5: {result['std'][25].item():.4f}")
            except Exception as e:
                print(f"  Error predicting task {example_task}: {e}")
    
    # Test multiple task predictions across different sizes
    sample_tasks = []
    for num_points in range(1, 11):
        tasks_with_n = task_predictor.get_tasks_by_num_points(num_points)
        if tasks_with_n:
            sample_tasks.append(tasks_with_n[0])  # Take first task of each size
    
    print(f"\nTesting multiple task predictions:")
    print(f"Sample tasks: {sample_tasks}")
    
    try:
        test_x = torch.linspace(0, 1, 50, device=device)
        multi_results = task_predictor.predict_multiple_tasks(
            sample_tasks, test_x, return_std=True
        )
        
        print(f"Successfully predicted {len([m for m in multi_results['means'] if m is not None])} tasks")
        for i, task_idx in enumerate(sample_tasks):
            if multi_results['means'][i] is not None:
                num_points = multi_results['num_train_points'][i]
                mean_val = multi_results['means'][i][25].item()
                std_val = multi_results['stds'][i][25].item()
                print(f"  Task {task_idx} ({num_points}pts): mean={mean_val:.4f}, std={std_val:.4f}")
    
    except Exception as e:
        print(f"Error in multiple task prediction: {e}")
    
    # Test feature extractor output
    print(f"\nDeep kernel feature analysis:")
    test_input = torch.linspace(0, 1, 10).unsqueeze(-1).to(device)
    shared_feature_extractor.to(device)
    shared_feature_extractor.eval()
    
    with torch.no_grad():
        features = shared_feature_extractor(test_input)
        print(f"  Input shape: {test_input.shape}")
        print(f"  Feature shape: {features.shape}")
        print(f"  Feature range: [{features.min().item():.3f}, {features.max().item():.3f}]")
        print(f"  Feature mean: {features.mean().item():.3f}")
    
    shared_feature_extractor.cpu()
    
    # Create visualization for different task sizes
    print(f"\nCreating visualization for different task sizes...")
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    plot_idx = 0
    for num_points in range(1, 11):
        tasks_with_n = task_predictor.get_tasks_by_num_points(num_points)
        if tasks_with_n and plot_idx < 10:
            ax = axes[plot_idx]
            task_idx = tasks_with_n[0]
            
            try:
                # Get predictions
                test_x = torch.linspace(0, 1, 50, device=device)
                result = task_predictor.predict_task(
                    task_idx, test_x, return_full_dist=True
                )
                
                # Generate training data for visualization
                train_x = torch.linspace(0, 1, num_points, device=device)
                train_x_np = train_x.cpu().numpy()
                
                # True training targets (with noise for visualization)
                torch.manual_seed(task_idx)  # Consistent noise per task
                true_train_y = (amplitudes[task_idx] * 
                               torch.sin(frequencies[task_idx] * train_x * 2 * math.pi + phases[task_idx]) +
                               torch.randn_like(train_x) * noise_levels[task_idx])
                
                # Plot training data
                ax.plot(train_x_np, true_train_y.cpu().numpy(), 'k*', markersize=8, label='Training')
                
                # Plot predictions
                test_x_np = test_x.cpu().numpy()
                ax.plot(test_x_np, result['mean'].cpu().numpy(), 'b-', linewidth=2, label='Deep GP')
                ax.fill_between(test_x_np, 
                               result['lower_ci'].cpu().numpy(), 
                               result['upper_ci'].cpu().numpy(), 
                               alpha=0.3, color='blue')
                
                # Plot true function
                true_y = (amplitudes[task_idx] * 
                         torch.sin(frequencies[task_idx] * test_x * 2 * math.pi + phases[task_idx]))
                ax.plot(test_x_np, true_y.cpu().numpy(), 'r--', linewidth=1, label='True')
                
                ax.set_title(f'Task {task_idx} (Deep Kernel)\n{num_points} training points')
                ax.grid(True, alpha=0.3)
                if plot_idx == 0:
                    ax.legend(fontsize=8)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {str(e)[:30]}...', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{num_points} points (Error)')
            
            plot_idx += 1
    
    plt.tight_layout()
    plt.savefig('deep_kernel_variable_size_gp_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Compute statistics
    print(f"\n=== Deep Kernel Training Statistics ===")
    avg_loss_by_size = defaultdict(list)
    avg_time_by_size = defaultdict(list)
    
    for stat in training_stats:
        avg_loss_by_size[stat['num_points']].append(stat['final_loss'])
        avg_time_by_size[stat['num_points']].append(stat['training_time'])
    
    print(f"Performance by number of training points:")
    for num_points in sorted(avg_loss_by_size.keys()):
        avg_loss = np.mean(avg_loss_by_size[num_points])
        avg_time = np.mean(avg_time_by_size[num_points])
        num_tasks = len(task_groups[num_points])
        print(f"  {num_points:2d} points: {num_tasks:5,} tasks, "
              f"avg loss: {avg_loss:.3f}, avg time/batch: {avg_time:.2f}s")
    
    total_batches = len(training_stats)
    avg_loss = np.mean([s['final_loss'] for s in training_stats])
    
    # Calculate total parameters
    gp_params = sum(sum(p.numel() for p in model.parameters()) for models in trained_models_by_size.values() for model in models)
    nn_params = sum(p.numel() for p in shared_feature_extractor.parameters())
    total_params = gp_params + nn_params
    
    print(f"\nOverall Statistics:")
    print(f"  Total tasks: {total_num_tasks:,}")
    print(f"  Total batches: {total_batches}")
    print(f"  Total training time: {total_training_time/60:.1f} minutes")
    print(f"  Average time per task: {total_training_time/total_num_tasks*1000:.1f} ms")
    print(f"  Average final loss: {avg_loss:.3f}")
    print(f"\nParameter Efficiency:")
    print(f"  Shared NN parameters: {nn_params:,}")
    print(f"  Total GP parameters: {gp_params:,}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Parameters per task: {total_params/total_num_tasks:.1f}")
    print(f"  Shared NN fraction: {nn_params/total_params*100:.1f}%")
    
    # Save results including the shared feature extractor
    save_data = {
        'shared_feature_extractor_state_dict': shared_feature_extractor.state_dict(),
        'shared_feature_extractor_config': {
            'input_dim': 1,
            'hidden_dim': nn_hidden_dim,
            'output_dim': nn_output_dim
        },
        'task_assignments': task_assignments,
        'batch_sizes_by_size': batch_sizes_by_size,
        'training_stats': training_stats,
        'config': {
            'total_tasks': total_num_tasks,
            'min_train_points': min_train_points,
            'max_train_points': max_train_points,
            'total_time': total_training_time,
            'avg_loss': avg_loss,
            'nn_params': nn_params,
            'gp_params': gp_params,
            'total_params': total_params
        },
        'task_parameters': {
            'amplitudes': amplitudes,
            'frequencies': frequencies,
            'phases': phases,
            'noise_levels': noise_levels,
            'num_train_points_per_task': num_train_points_per_task
        }
    }
    
    torch.save(save_data, 'deep_kernel_variable_size_training_results.pt')
    
    print(f"\nDeep kernel training completed successfully!")
    print(f"Results saved to deep_kernel_variable_size_training_results.pt")
    
    print(f"\n=== Usage Examples ===")
    print(f"# Predict a specific task:")
    print(f"result = task_predictor.predict_task(12345, test_x, return_std=True)")
    print(f"print(f'Task has {{result[\"num_train_points\"]}} training points')")
    print(f"")
    print(f"# Get tasks by number of training points:")
    print(f"one_point_tasks = task_predictor.get_tasks_by_num_points(1)")
    print(f"ten_point_tasks = task_predictor.get_tasks_by_num_points(10)")
    print(f"")
    print(f"# Get task information:")
    print(f"info = task_predictor.get_task_info(12345)")
    print(f"print(f'Task has {{info[\"num_train_points\"]}} training points')")
    print(f"")
    print(f"# The shared neural network enables:")
    print(f"#   - Parameter sharing across all {total_num_tasks:,} tasks")
    print(f"#   - Only {nn_params:,} shared NN parameters vs {gp_params:,} GP parameters")
    print(f"#   - Better feature representation through deep kernel learning")
    
    return task_predictor

if __name__ == "__main__":
    predictor = main()
