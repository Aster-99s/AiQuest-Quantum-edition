import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

class CellularTowerOptimizer:
    def __init__(self, data_path=None, output_dir='output'):
        """
        Initialize the Cellular Tower Optimizer
        
        Parameters:
        data_path (str): Path to CSV file containing tower data (optional)
        output_dir (str): Directory to save output files
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize preprocessing components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        
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
        
        # Group data by region for region-specific analysis
        self.regions = self.data['region'].unique()
        self.region_data = {region: self.data[self.data['region'] == region] for region in self.regions}
    
    def build_model(self):
        """Build a simpler neural network model for bandwidth prediction"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu', input_shape=(self.X.shape[1],)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1)  # Output layer for bandwidth prediction
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.model = model
        return model
    
    def train_model(self, epochs=100, batch_size=64, validation_split=0.2):
        """Train the neural network model"""
        if self.model is None:
            self.build_model()
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=validation_split, random_state=42
        )
        
        # Add early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Plot and save training history
        self._plot_training_history(history)
        
        return history
    
    def _plot_training_history(self, history):
        """Plot and save training history"""
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot MAE
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_history.png'))
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
            min_dist_to_existing = min(
                np.sqrt(np.sum((existing_towers - center)**2, axis=0))
            )
            
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
                        dist_to_existing = min(
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
        if self.model is None:
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
        predictions = self.model.predict(input_scaled)
        
        return predictions.ravel()
    
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
        with open(os.path.join(self.output_dir, 'tower_recommendations.json'), 'w') as f:
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
            plt.title(f'Tower Distribution in {region}')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.grid(alpha=0.3)
            plt.legend()
            
            # Save figure
            plt.savefig(os.path.join(self.output_dir, f'tower_map_{region}.png'))
            plt.close()

# Example usage
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

if __name__ == "__main__":
    # Generate synthetic data with realistic patterns
    data = generate_synthetic_data(num_regions=5, towers_per_region=60)
    
    # Create optimizer
    optimizer = CellularTowerOptimizer(output_dir='tower_optimization_results')
    optimizer.load_data(data)
    optimizer.preprocess_data()
    
    # Train the model
    optimizer.build_model()
    optimizer.train_model(epochs=50)
    
    # Generate recommendations
    recommendations = optimizer.recommend_tower_placements(towers_per_region=6)
    
    # Visualize results
    optimizer.visualize_recommendations(recommendations)
    
    print("Tower optimization complete. Results saved to 'tower_optimization_results' directory.")