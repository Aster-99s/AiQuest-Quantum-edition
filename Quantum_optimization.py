import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

class QuantumCellularTowerOptimizer:
    def __init__(self, data_path=None, output_dir='output'):
        """
        Initialize the Quantum Cellular Tower Optimizer
        
        Parameters:
        data_path (str): Path to CSV file containing tower data (optional)
        output_dir (str): Directory to save output files
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize preprocessing components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Initialize QML-related attributes
        self.n_qubits = 4  # Will be updated based on features
        self.n_layers = 2
        self.dev = None
        self.qnode = None
        self.weights = None
        
        # Load data if path provided
        if data_path:
            self.data = pd.read_csv(data_path)
        else:
            self.data = None
    
    def load_data(self, data):
        """Load data from a pandas DataFrame"""
        self.data = data
    
    def preprocess_data(self):
        """Preprocess the data for analysis and modeling"""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        # Encode regions
        self.data['region_encoded'] = self.label_encoder.fit_transform(self.data['region'])
        
        # Extract features and target
        features = self.data[['latitude', 'longitude', 'region_encoded']]
        self.X = self.scaler.fit_transform(features)  # Scaled features
        self.y = self.data['Tower_bandwidth'].values.reshape(-1, 1)
        
        # Set number of qubits based on feature count
        self.n_qubits = self.X.shape[1]
        
        # Group data by region for region-specific analysis
        self.regions = self.data['region'].unique()
        self.region_data = {region: self.data[self.data['region'] == region] for region in self.regions}
    
    def _quantum_circuit(self, inputs, weights):
        """Define the quantum circuit architecture"""
        # Encode inputs into the quantum circuit
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)
        
        # Apply parameterized quantum layers
        for layer in range(self.n_layers):
            # Entangling layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[self.n_qubits - 1, 0])  # Connect last qubit to first
            
            # Rotation layer with weights
            for i in range(self.n_qubits):
                qml.RX(weights[layer, i, 0], wires=i)
                qml.RY(weights[layer, i, 1], wires=i)
                qml.RZ(weights[layer, i, 2], wires=i)

        # Measure expectation value of Z on first qubit as output
        return qml.expval(qml.PauliZ(0))
    
    def build_model(self):
        """Build the quantum machine learning model"""
        # Initialize quantum device
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        
        # Create QNode
        self.qnode = qml.QNode(self._quantum_circuit, self.dev)
        
        # Initialize weights randomly
        weight_shapes = {"weights": (self.n_layers, self.n_qubits, 3)}
        self.weights = np.random.uniform(0, 2*np.pi, size=(self.n_layers, self.n_qubits, 3))
        
        return self.qnode
    
    def _cost(self, weights, X_batch, y_batch):
        """Cost function for optimization"""
        predictions = np.array([self.qnode(x, weights) for x in X_batch])
        
        # Scale predictions to match the range of tower bandwidth
        scaled_predictions = (predictions + 1) * 125  # Maps [-1,1] to [0,250]
        
        # Mean squared error loss
        loss = np.mean((scaled_predictions - y_batch.flatten()) ** 2)
        return loss
    
    def _predict_batch(self, X):
        """Make predictions for a batch of inputs"""
        if self.weights is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        predictions = np.array([self.qnode(x, self.weights) for x in X])
        # Scale predictions from [-1,1] to realistic bandwidth values
        scaled_predictions = (predictions + 1) * 125  # Maps to [0,250]
        return scaled_predictions
    
    def train_model(self, epochs=20, batch_size=8):
        """Train the quantum model"""
        if self.qnode is None:
            self.build_model()
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Prepare for training history tracking
        train_losses = []
        val_losses = []
        
        # Use simpler optimizer for QML
        opt = qml.GradientDescentOptimizer(stepsize=0.1)
        
        print("Training quantum model...")
        for epoch in range(epochs):
            # Process in small batches due to quantum simulation overhead
            batch_indices = np.random.choice(len(X_train), min(batch_size, len(X_train)), replace=False)
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            # Update weights
            self.weights = opt.step(lambda w: self._cost(w, X_batch, y_batch), self.weights)
            
            # Compute training and validation loss
            if epoch % 2 == 0:  # Check less frequently to speed up training
                train_loss = self._cost(self.weights, X_train[:min(100, len(X_train))], y_train[:min(100, len(X_train))])
                val_loss = self._cost(self.weights, X_val[:min(100, len(X_val))], y_val[:min(100, len(X_val))])
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                print(f"Epoch {epoch}: Train Loss = {train_loss:.2f}, Val Loss = {val_loss:.2f}")
        
        # Plot and save training history
        self._plot_training_history(train_losses, val_losses)
        
        return train_losses, val_losses
    
    def _plot_training_history(self, train_losses, val_losses):
        """Plot and save training history"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Quantum Model Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'quantum_training_history.png'))
        plt.close()
    
    def identify_high_load_areas(self, region, percentile=75):
        """Identify areas with high bandwidth demand in a region"""
        region_df = self.region_data[region]
        
        # Get towers with high bandwidth usage
        threshold = np.percentile(region_df['Tower_bandwidth'], percentile)
        high_load_towers = region_df[region_df['Tower_bandwidth'] >= threshold]
        
        return high_load_towers
    
    def analyze_demand_clusters(self, region, n_clusters=5):
        """
        Analyze demand clusters in a region using K-means clustering
        weighted by bandwidth usage
        """
        high_load_towers = self.identify_high_load_areas(region)
        
        if len(high_load_towers) < n_clusters:
            n_clusters = max(1, len(high_load_towers) // 2)
        
        # Use tower coordinates
        tower_locations = high_load_towers[['latitude', 'longitude']].values
        bandwidth_weights = high_load_towers['Tower_bandwidth'].values
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(tower_locations, sample_weight=bandwidth_weights)
        
        # Get cluster centers and analyze density
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        # Calculate demand score for each cluster
        demand_scores = []
        for i in range(n_clusters):
            cluster_towers = high_load_towers.iloc[labels == i]
            total_bandwidth = cluster_towers['Tower_bandwidth'].sum()
            tower_count = len(cluster_towers)
            # Score is a combination of total bandwidth and tower density
            demand_scores.append({
                'center': centers[i],
                'total_bandwidth': total_bandwidth,
                'tower_count': tower_count,
                'demand_score': total_bandwidth / max(1, tower_count)  # Demand per tower
            })
        
        # Sort by demand score (higher is better)
        demand_scores.sort(key=lambda x: x['demand_score'], reverse=True)
        
        return demand_scores
    
    def generate_optimal_locations(self, region, n_towers=3, min_distance=0.03):
        """
        Generate optimal new tower locations based on demand clusters and
        keeping minimum distance from existing towers
        """
        region_df = self.region_data[region]
        existing_towers = region_df[['latitude', 'longitude']].values
        
        # Get demand clusters
        demand_clusters = self.analyze_demand_clusters(region)
        
        optimal_locations = []
        for cluster_info in demand_clusters:
            # Start with the cluster center
            center = cluster_info['center']
            
            # Check if it's far enough from existing towers
            distances = np.sqrt(np.sum((existing_towers - center)**2, axis=1))
            min_dist_to_existing = np.min(distances)
            
            if min_dist_to_existing > min_distance:
                # This cluster center is a good candidate
                optimal_locations.append(center)
            else:
                # Need to find a nearby location that's far enough from existing towers
                # Create a grid around the center
                grid_size = 10
                radius = min_distance * 2
                
                lat_offsets = np.linspace(-radius, radius, grid_size)
                lon_offsets = np.linspace(-radius, radius, grid_size)
                
                for lat_offset in lat_offsets:
                    for lon_offset in lon_offsets:
                        candidate = center + np.array([lat_offset, lon_offset])
                        dist_to_existing = np.min(
                            np.sqrt(np.sum((existing_towers - candidate)**2, axis=1))
                        )
                        
                        if dist_to_existing > min_distance:
                            optimal_locations.append(candidate)
                            break
                    else:
                        continue
                    break
            
            # Stop if we have enough locations
            if len(optimal_locations) >= n_towers:
                break
                
        return np.array(optimal_locations[:n_towers])
    
    def predict_tower_demand(self, locations, region):
        """Predict bandwidth demand for new tower locations"""
        if self.weights is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Prepare input features
        region_encoded = self.label_encoder.transform([region])[0]
        
        # Create input array with region encoding
        input_data = np.zeros((len(locations), 3))
        input_data[:, 0:2] = locations  # lat, lon
        input_data[:, 2] = region_encoded
        
        # Scale features
        input_scaled = self.scaler.transform(input_data)
        
        # Make predictions
        predictions = self._predict_batch(input_scaled)
        
        return predictions
    
    def recommend_tower_placements(self, towers_per_region=3):
        """Generate tower placement recommendations for all regions"""
        recommendations = {}
        
        for region in self.regions:
            print(f"Processing region: {region}")
            
            # Generate optimal locations
            optimal_locations = self.generate_optimal_locations(
                region, 
                n_towers=towers_per_region
            )
            
            if len(optimal_locations) == 0:
                print(f"No suitable locations found for {region}")
                recommendations[region] = []
                continue
            
            # Predict demand for these locations
            predicted_demand = self.predict_tower_demand(optimal_locations, region)
            
            # Store recommendations
            recommendations[region] = {
                'locations': optimal_locations.tolist(),
                'predicted_bandwidth': predicted_demand.tolist()
            }
        
        # Save to file
        with open(os.path.join(self.output_dir, 'quantum_tower_recommendations.json'), 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        return recommendations
    
    def visualize_recommendations(self, recommendations):
        """Visualize existing towers and recommended placements"""
        for region in self.regions:
            region_df = self.region_data[region]
            
            # Create figure
            plt.figure(figsize=(10, 8))
            
            # Plot existing towers
            sc = plt.scatter(
                region_df['longitude'],
                region_df['latitude'],
                c=region_df['Tower_bandwidth'],
                cmap='viridis',
                alpha=0.7,
                s=30,
                label='Existing towers'
            )
            
            # Plot recommended towers if available
            if region in recommendations and recommendations[region]:
                locations = np.array(recommendations[region]['locations'])
                predicted_bw = np.array(recommendations[region]['predicted_bandwidth'])
                
                plt.scatter(
                    locations[:, 1],  # Longitude
                    locations[:, 0],  # Latitude
                    marker='X',
                    c='red',
                    s=100,
                    label='Recommended new towers'
                )
            
            # Add colorbar for existing tower bandwidth
            cbar = plt.colorbar(sc)
            cbar.set_label('Bandwidth (Mbps)')
            
            # Add labels and title
            plt.title(f'Tower Distribution in {region} - Quantum Model')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.grid(alpha=0.3)
            plt.legend()
            
            # Save figure
            plt.savefig(os.path.join(self.output_dir, f'quantum_tower_map_{region}.png'))
            plt.close()

# Function to generate synthetic data (same as original)
def generate_synthetic_data(num_regions=3, towers_per_region=30):
    """Generate more realistic synthetic data with bandwidth hotspots"""
    regions = [f"Region_{i+1}" for i in range(num_regions)]
    data = []
    
    for region_idx, region in enumerate(regions):
        # Define region center
        center_lat = 34.0 + region_idx * 1.5
        center_lon = -118.0 - region_idx * 1.5
        
        # Create a few hotspots in each region
        num_hotspots = np.random.randint(2, 5)
        hotspots = []
        
        for _ in range(num_hotspots):
            # Hotspot location with some distance from center
            hotspot_lat = center_lat + np.random.uniform(-0.2, 0.2)
            hotspot_lon = center_lon + np.random.uniform(-0.2, 0.2)
            hotspot_intensity = np.random.uniform(150, 250)  # Higher bandwidth
            hotspots.append((hotspot_lat, hotspot_lon, hotspot_intensity))
        
        for i in range(towers_per_region):
            # Randomly choose whether this tower is near a hotspot
            if np.random.random() < 0.6:  # 60% chance of being near a hotspot
                # Pick a random hotspot
                hotspot = hotspots[np.random.randint(0, len(hotspots))]
                hotspot_lat, hotspot_lon, hotspot_intensity = hotspot
                
                # Place tower near the hotspot
                distance = np.random.exponential(0.1)  # Closer to hotspot is more likely
                angle = np.random.uniform(0, 2 * np.pi)
                
                lat = hotspot_lat + distance * np.cos(angle)
                lon = hotspot_lon + distance * np.sin(angle)
                
                # Bandwidth decreases with distance from hotspot
                bandwidth = hotspot_intensity * np.exp(-distance * 10) + np.random.normal(0, 10)
            else:
                # Random tower not near a hotspot
                lat = center_lat + np.random.normal(0, 0.15)
                lon = center_lon + np.random.normal(0, 0.15)
                bandwidth = np.random.uniform(30, 80)  # Lower bandwidth
            
            tower_id = f"{region}_tower_{i+1}"
            data.append([
                region, tower_id, lat, lon, max(20, bandwidth)  # Ensure positive bandwidth
            ])
    
    df = pd.DataFrame(data, columns=[
        'region', 'tower_id', 'latitude', 'longitude', 'Tower_bandwidth'
    ])
    
    return df

# Example usage
if __name__ == "__main__":
    # Generate synthetic data with realistic patterns
    print("Generating synthetic data...")
    data = generate_synthetic_data(num_regions=3, towers_per_region=40)  # Reduced for faster quantum simulation
    
    # Create optimizer
    print("Initializing quantum optimizer...")
    optimizer = QuantumCellularTowerOptimizer(output_dir='quantum_tower_optimization_results')
    optimizer.load_data(data)
    optimizer.preprocess_data()
    
    # Train the model (fewer epochs for quantum simulation)
    print("Training quantum model...")
    optimizer.build_model()
    optimizer.train_model(epochs=15, batch_size=8)
    
    # Generate recommendations
    print("Generating tower placement recommendations...")
    recommendations = optimizer.recommend_tower_placements(towers_per_region=4)
    
    # Visualize results
    print("Visualizing results...")
    optimizer.visualize_recommendations(recommendations)
    
    print("Quantum tower optimization complete. Results saved to 'quantum_tower_optimization_results' directory.")