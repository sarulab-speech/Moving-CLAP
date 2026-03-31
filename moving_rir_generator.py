import numpy as np
import torch
import gpuRIR
import os
from tqdm import tqdm
import pickle
from scipy import signal
from itertools import product
import argparse

class MovingRIRGenerator:
    def __init__(self, 
                 room_sizes=None, 
                 absorptions=None,
                 fs=16000,
                 num_segments=100):
        
        self.fs = fs
        self.num_segments = num_segments
        
        # gpuRIR settings
        gpuRIR.activateMixedPrecision(False)
        gpuRIR.activateLUT(True)
        
        # Default room settings
        if room_sizes is None:
            self.room_sizes = [
                [6.0, 5.0, 3.0],
                [8.0, 6.0, 3.5], 
                [10.0, 8.0, 4.0],
            ]
        else:
            self.room_sizes = room_sizes
            
        # Default absorption settings
        if absorptions is None:
            self.absorptions = [0.3, 0.4, 0.5]
        else:
            self.absorptions = absorptions

        # Define zones for DOA sampling
        self.doa_zones = {
            "left side": (-1.0, -0.6),       #(-90° ~ -54°)
            "front-left": (-0.6, -0.2),      # (-54° ~ -18°)
            "front": (-0.2, 0.2),            # (-18° ~ 18°)
            "front-right": (0.2, 0.6),       # (18° ~ 54°)
            "right side": (0.6, 1.0)         # (54° ~ 90°)
        }
        self.zone_names = list(self.doa_zones.keys())
        
        # Initialize random seed
        np.random.seed(42)
    
    def generate_trajectory(self, room_size, trajectory_type="linear", start_zone=None, end_zone=None):
        # Select zones randomly
        if start_zone is None:
            start_zone = np.random.choice(self.zone_names)
        if end_zone is None:
            end_zone = np.random.choice(self.zone_names)
        # Stationary determination (within the same zone)
        is_stationary = (start_zone == end_zone)

        if is_stationary:
            # If stationary, limit the DOA change to a small range within the zone
            zone_min, zone_max = self.doa_zones[start_zone]
            start_doa = np.random.uniform(zone_min + 0.089, zone_max - 0.089)
            max_change = 0.056 # about 5 degrees
            safe_min = zone_min + 0.089
            safe_max = zone_max - 0.089
            
            valid_min = max(safe_min, start_doa - max_change)
            valid_max = min(safe_max, start_doa + max_change)
            
            if valid_max > valid_min:
                end_doa = np.random.uniform(valid_min, valid_max)
            else:
                end_doa = start_doa
        else:
            doa_change_threshold = 0.22  # Minimum change for moving sources
            doa_change = 0.0
            while doa_change < doa_change_threshold:
                start_doa = np.random.uniform(
                    self.doa_zones[start_zone][0] + 0.089, 
                    self.doa_zones[start_zone][1] - 0.089
                )
                end_doa = np.random.uniform(
                    self.doa_zones[end_zone][0] + 0.089, 
                    self.doa_zones[end_zone][1] - 0.089
                )
                doa_change = abs(end_doa - start_doa)

        # Generate trajectory based on type
        if trajectory_type == "linear":
            return self._generate_linear_trajectory(room_size, start_doa, end_doa, is_stationary, start_zone, end_zone)
        elif trajectory_type == "circular":
            return self._generate_circular_trajectory(room_size, start_doa, end_doa, is_stationary, start_zone, end_zone)
        else:
            raise ValueError(f"Unknown trajectory type: {trajectory_type}")
        
    def calculate_y_limit(self, start_doa, end_doa, room_size, dist_to_left=None, dist_to_right=None):
        start_angle_rad = np.radians(start_doa * 90.0)
        end_angle_rad = np.radians(end_doa * 90.0)
        
        max_y_start = float('inf')
        if abs(np.tan(start_angle_rad)) > 1e-6:
            if start_doa < 0:
                max_y_start = dist_to_left / abs(np.tan(start_angle_rad))
            else:
                max_y_start = dist_to_right / abs(np.tan(start_angle_rad))
        
        max_y_end = float('inf')
        if abs(np.tan(end_angle_rad)) > 1e-6:
            if end_doa < 0:
                max_y_end = dist_to_left / abs(np.tan(end_angle_rad))
            else:
                max_y_end = dist_to_right / abs(np.tan(end_angle_rad))
        return min(max_y_start, max_y_end)

    
    def _generate_linear_trajectory(self, room_size, start_doa, end_doa, is_stationary, start_zone, end_zone):
        """
        Create a linear trajectory from start_doa to end_doa
        """
        margin = 0.5
        height = 1.5
        mic_x = room_size[0] / 2
        mic_y = room_size[1] / 2
        
        min_dim = min(room_size[0], room_size[1])
        
        dist_to_left = mic_x - margin
        dist_to_right = (room_size[0] - mic_x) - margin
        
        min_y = 1
        limit_y_by_room = (room_size[1] / 2) - margin
        global_max_y = min(self.calculate_y_limit(start_doa, end_doa, room_size, dist_to_left, dist_to_right), limit_y_by_room)
        # Until a valid y_distance is found
        actual_y_distance = None
        while actual_y_distance is None:
            if global_max_y >= min_y:
                actual_y_distance = np.random.uniform(min_y, global_max_y)
            else:
                if is_stationary:
                    zone_min, zone_max = self.doa_zones[start_zone]
                    start_doa = np.random.uniform(zone_min + 0.089, zone_max - 0.089)
                    max_change = 0.056 # about 5 degrees
                    safe_min = zone_min + 0.089
                    safe_max = zone_max - 0.089
                    valid_min = max(safe_min, start_doa - max_change)
                    valid_max = min(safe_max, start_doa + max_change)
                    if valid_max > valid_min:
                        end_doa = np.random.uniform(valid_min, valid_max)
                    else:
                        end_doa = start_doa
                else:
                    doa_change_threshold = 0.22  # Minimum change for moving sources
                    doa_change = 0.0
                    while doa_change < doa_change_threshold:
                        start_doa = np.random.uniform(
                            self.doa_zones[start_zone][0] + 0.089, 
                            self.doa_zones[start_zone][1] - 0.089
                        )
                        end_doa = np.random.uniform(
                            self.doa_zones[end_zone][0] + 0.089, 
                            self.doa_zones[end_zone][1] - 0.089
                        )
                        doa_change = abs(end_doa - start_doa)
                global_max_y = min(self.calculate_y_limit(start_doa, end_doa, room_size, dist_to_left, dist_to_right), limit_y_by_room)

        line_y = mic_y + actual_y_distance
        start_angle_rad = np.radians(start_doa * 90.0)
        end_angle_rad = np.radians(end_doa * 90.0)
        start_x = mic_x + actual_y_distance * np.tan(start_angle_rad)
        end_x = mic_x + actual_y_distance * np.tan(end_angle_rad)
        start_pos = np.array([start_x, line_y, height])
        end_pos = np.array([end_x, line_y, height])
               
        positions = np.array([
            np.linspace(start_pos[i], end_pos[i], self.num_segments)
            for i in range(3)
        ]).T

        tol = 1e-6
        # check DOA within zones
        assert self.doa_zones[start_zone][0] - tol <= start_doa <= self.doa_zones[start_zone][1] + tol, \
            f"Start DOA {start_doa} out of zone {start_zone}"
        assert self.doa_zones[end_zone][0] - tol <= end_doa <= self.doa_zones[end_zone][1] + tol, \
            f"End DOA {end_doa} out of zone {end_zone}"
        
        timing_info = {
            'is_stationary': is_stationary,
            'start_zone': start_zone,
            'end_zone': end_zone,
            'doa_change': abs(end_doa - start_doa),
            'trajectory_type': 'linear',
            'y_distance': actual_y_distance
        }
        
        return positions, start_pos, end_pos, start_doa, end_doa, is_stationary, timing_info
    
    def _generate_circular_trajectory(self, room_size, start_doa, end_doa, is_stationary, start_zone, end_zone):
        """
        Create a circular trajectory from start_doa to end_doa
        """
        center = np.array([room_size[0]/2, room_size[1]/2, 1.5])
        distance = min(room_size[0], room_size[1]) * np.random.uniform(0.2, 0.4)
        
        doa_values = np.linspace(start_doa, end_doa, self.num_segments)
        
        positions = []
        for doa in doa_values:
            azimuth_deg = doa * 90.0
            azimuth_rad = np.radians(azimuth_deg)
            
            x = center[0] + distance * np.sin(azimuth_rad)
            y = center[1] + distance * np.cos(azimuth_rad)
            z = center[2]
            
            # check boundaries
            margin = 0.5
            x = np.clip(x, margin, room_size[0] - margin)
            y = np.clip(y, margin, room_size[1] - margin)
            
            positions.append([x, y, z])
        
        positions = np.array(positions)
        
        timing_info = {
            'is_stationary': is_stationary,
            'start_zone': start_zone,
            'end_zone': end_zone,
            'doa_change': abs(end_doa - start_doa),
            'trajectory_type': 'circular',
            'y_distance': distance
        }
        
        return positions, positions[0], positions[-1], start_doa, end_doa, is_stationary, timing_info
        
    def generate_microphone_positions(self, room_size):
        center = np.array([room_size[0]/2, room_size[1]/2, 1.5])
        mic_distance =  16.98e-2
        
        positions = np.array([
            center + np.array([-mic_distance/2, 0, 0]),  # left
            center + np.array([mic_distance/2, 0, 0])    # right
        ])
        
        return positions

    def calculate_t60(self, room_size, absorption):
        surface_area = 2 * (room_size[0]*room_size[1] + 
                        room_size[1]*room_size[2] + 
                        room_size[2]*room_size[0])
        volume = np.prod(room_size)
        A = surface_area * absorption
        return 0.161 * volume / A
    
    def calculate_trajectory_rir(self, room_size, absorption, trajectory_positions, mic_positions):
        # Prepare parameters
        T60 = self.calculate_t60(room_size, absorption)
        abs_weights = [0.9] * 6
        beta = gpuRIR.beta_SabineEstimation(room_size, T60, abs_weights=abs_weights)
        Tdiff = gpuRIR.att2t_SabineEstimator(15.0, T60)
        Tmax = gpuRIR.att2t_SabineEstimator(60.0, T60)
        nb_img = gpuRIR.t2n(Tdiff, room_size)
        
        # Calculate RIR
        RIRs = gpuRIR.simulateRIR(
            room_size, beta, trajectory_positions, mic_positions, 
            nb_img, Tmax, self.fs, Tdiff=Tdiff
        )
        
        # High-pass filter to remove DC and low-frequency noise
        sos = signal.butter(4, 50, 'hp', fs=self.fs, output='sos')

        # RIRs shape: (num_trajectory_points, num_mics, time)
        RIRs = signal.sosfilt(sos, RIRs, axis=-1).astype(np.float32)
        
        return RIRs
    
    def generate_trajectory_labels(self, trajectory_positions, mic_positions):
        """Generate labels for the moving trajectory"""
        mic_center = mic_positions.mean(axis=0)
        
        labels = []
        for pos in trajectory_positions:
            # Relative position from microphone center
            relative_pos = pos - mic_center
            
            # Calculate DOA (azimuth angle)
            # y-axis is forward, x-axis is left-right
            azimuth = np.arctan2(relative_pos[0], relative_pos[1])
            azimuth_deg = np.degrees(azimuth)
            azimuth_deg = np.clip(azimuth_deg, -90.0, 90.0)
            
            doa_normalized = azimuth_deg / 90.0
            
            # 距離計算
            distance = np.linalg.norm(relative_pos[:2])
            
            labels.append({
                'doa': doa_normalized,
                'azimuth_deg': azimuth_deg,
                'distance': distance,
                'position': pos.copy(),
                'relative_x': relative_pos[0]
            })
        
        return labels
    
    def generate_dataset(self, output_base_dir, split_names=['train', 'val', 'test'],
                        num_room_conditions_train=None,
                        num_room_conditions_val=None,
                        num_room_conditions_test=None):
        # Prepare splits configuration
        splits_config = []
        for split in split_names:
            if split == 'train':
                num_conditions = num_room_conditions_train
            elif split == 'val':
                num_conditions = num_room_conditions_val
            elif split == 'test':
                num_conditions = num_room_conditions_test
            else:
                print(f"Unknown split name: {split}, skipping...")
                continue
            splits_config.append((split, num_conditions))
        
        for split_name, num_conditions in splits_config:
            if num_conditions is not None and num_conditions > 0:
                print(f"\nGenerating {split_name} split...")
                self._generate_split_unified(output_base_dir, split_name, num_conditions)
    
    def _generate_split_unified(self, output_base_dir, split_name, num_room_conditions):
        output_dir = os.path.join(output_base_dir, split_name)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'rir'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
        
        metadata = []
        global_idx = 0
        
        trajectory_types = ['linear', 'circular']
        room_params = list(product(self.room_sizes, self.absorptions))
        num_params = len(room_params)
        
        for room_condition_id in tqdm(range(num_room_conditions), desc=f"{split_name} room conditions"):
            # Select room condition, cycling through all combinations in order
            room_size, absorption = room_params[room_condition_id % num_params]
            mic_positions = self.generate_microphone_positions(room_size)
            
            # 25 zone combinations
            for start_zone in self.zone_names:
                for end_zone in self.zone_names:
                    for traj_type in trajectory_types:
                        # Generate trajectory
                        trajectory_positions, start_pos, end_pos, start_doa, end_doa, is_stationary, timing_info = \
                            self.generate_trajectory(room_size, traj_type, start_zone=start_zone, end_zone=end_zone)

                        # Calculate RIRs
                        RIRs = self.calculate_trajectory_rir(room_size, absorption, trajectory_positions, mic_positions)
                        
                        # Generate labels
                        labels = self.generate_trajectory_labels(trajectory_positions, mic_positions)
                        
                        # Save RIR and labels
                        rir_file = os.path.join(output_dir, 'rir', f'{global_idx:06d}.npy')
                        label_file = os.path.join(output_dir, 'labels', f'{global_idx:06d}.pkl')
                        
                        np.save(rir_file, RIRs)
                        with open(label_file, 'wb') as f:
                            pickle.dump(labels, f)
                        
                        metadata.append({
                            'rir_id': global_idx,
                            'room_condition_id': room_condition_id,
                            'room_size': room_size,
                            'absorption': absorption,
                            'trajectory_type': traj_type,
                            'trajectory_positions': trajectory_positions,
                            'start_zone': start_zone,
                            'end_zone': end_zone,
                            'start_doa': start_doa,
                            'end_doa': end_doa,
                            'is_stationary': is_stationary,
                            'doa_change': timing_info['doa_change'],
                            'y_distance': timing_info['y_distance'],
                            'num_segments': self.num_segments,
                        })
                        
                        global_idx += 1
        
        metadata_file = os.path.join(output_dir, 'metadata.pkl')
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"  Generated {global_idx} RIRs for {split_name} ({num_room_conditions} room conditions)")
        print(f"  Metadata saved: {metadata_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='data/moving_rir_dataset',
                       help='Output base directory')
    parser.add_argument('--num_room_conditions_train', type=int, default=1000,
                       help='Number of room conditions for training (50 RIRs each)')
    parser.add_argument('--num_room_conditions_val', type=int, default=10,
                       help='Number of room conditions for validation (50 RIRs each)')
    parser.add_argument('--num_room_conditions_test', type=int, default=0,
                       help='Number of room conditions for test (50 RIRs each)')
    args = parser.parse_args()
    generator = MovingRIRGenerator()
    generator.generate_dataset(
        output_base_dir=args.output_dir,
        num_room_conditions_train=args.num_room_conditions_train,
        num_room_conditions_val=args.num_room_conditions_val,
        num_room_conditions_test=args.num_room_conditions_test
    )