import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
import pandas as pd
import json
import rioxarray

class FireDataset(Dataset):
    def __init__(self, data_dir, sequence_length=3, transform=None):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.samples = []
        self._load_landscape_data()
        self._load_weather_data()
        self._load_spatial_index()
        self._prepare_samples()

    def _load_spatial_index(self):
        indices_path = os.path.join(self.data_dir, 'landscape', 'indices.json')
        with open(indices_path, 'r') as f:
            self.indices = json.load(f)

    def _load_landscape_data(self):
        landscape_path = os.path.join(self.data_dir, 'landscape', 'Input_Geotiff.tif')
        # Load landscape data using rioxarray
        with rioxarray.open_rasterio(landscape_path) as src:
            self.landscape_data = src.where(src != -9999.0, -1)
            self.landscape_max = self.landscape_data.max(dim=["x", "y"]).values
            self.landscape_min = self.landscape_data.min(dim=["x", "y"]).values
        # Normalize the landscape data
        for i in range(len(self.landscape_max)):
            self.landscape_data[i, :, :] = (self.landscape_data[i, :, :] - self.landscape_min[i]) / (self.landscape_max[i] - self.landscape_min[i])

    def _load_weather_data(self):
        # Load weather history
        weather_history_path = os.path.join(self.data_dir, 'landscape', 'WeatherHistory.csv')
        self.weather_history = pd.read_csv(weather_history_path, header=None)

        # Load individual weather files
        self.weathers = {}
        weather_folder = os.path.join(self.data_dir, 'landscape', 'Weathers')
        weather_files = os.listdir(weather_folder)

        # Variables to store global max and min for normalization
        self.max_ws = float('-inf')
        self.min_ws = float('inf')
        self.max_wd = float('-inf')
        self.min_wd = float('inf')

        # Load weather data and find global max and min
        for weather_file in weather_files:
            weather_file_path = os.path.join(weather_folder, weather_file)
            weather_df = pd.read_csv(weather_file_path)
            self.weathers[weather_file] = weather_df

            # Update global max and min
            self.max_ws = max(self.max_ws, weather_df['WS'].max())
            self.min_ws = min(self.min_ws, weather_df['WS'].min())
            self.max_wd = max(self.max_wd, weather_df['WD'].max())
            self.min_wd = min(self.min_wd, weather_df['WD'].min())

    def _get_spatial_indices(self, sample):
        # Placeholder function to get spatial indices
        # You need to define how to get y, y_, x, x_ based on the sample
        # For example, you might have an indices.json file similar to MyDatasetV2
        # Here is an example:
        seq_id = sample['sequence_id']
        seq_id = str(int(seq_id))
        indices = self.indices.get(seq_id)
        if indices:
            y, y_, x, x_ = indices
        else:
            # If indices are not available, use default values or raise an error
            raise ValueError(f'Spatial indices not found for sequence {seq_id}')
        return y, y_, x, x_

    def _get_weather_data(self, sample):
        # Get the sequence ID to retrieve the corresponding weather data
        seq_id = sample['sequence_id']

        # Get the weather file name from the weather history
        # Adjust this according to how your weather history is structured
        weather_file_name = self.weather_history.iloc[int(seq_id) - 1].values[0].split("Weathers/")[1]
        weather_df = self.weathers[weather_file_name]

        # Get the weather data corresponding to the last fire frame in the sequence
        scenario_n = sample['fire_frame_indices'][-1]
        wind_speed = weather_df.iloc[scenario_n]['WS']
        wind_direction = weather_df.iloc[scenario_n]['WD']

        # Normalize weather data
        wind_speed_norm = (wind_speed - self.min_ws) / (self.max_ws - self.min_ws)
        wind_direction_norm = (wind_direction - self.min_wd) / (self.max_wd - self.min_wd)

        weather_tensor = torch.tensor([wind_speed_norm, wind_direction_norm], dtype=torch.float32)

        return weather_tensor


    def _prepare_samples(self):
        fire_frames_root = os.path.join(self.data_dir, 'fire_frames')
        isochrones_root = os.path.join(self.data_dir, 'isochrones')
        sequence_dirs = sorted(os.listdir(fire_frames_root))

        for seq_dir in sequence_dirs:
            seq_id = seq_dir.replace('sequence_', '')
            fire_seq_path = os.path.join(fire_frames_root, seq_dir)
            iso_seq_path = os.path.join(isochrones_root, seq_dir)

            # Get sorted list of fire frame files and isochrone files
            fire_frame_files = sorted([f for f in os.listdir(fire_seq_path) if f.endswith('.png')])
            iso_frame_files = sorted([f for f in os.listdir(iso_seq_path) if f.endswith('.png')])

            num_fire_frames = len(fire_frame_files)
            num_iso_frames = len(iso_frame_files)

            # Adjust num_samples to prevent index out of range
            num_samples = min(num_fire_frames - self.sequence_length, num_iso_frames - self.sequence_length - 1)
            num_samples = min(
                num_fire_frames - self.sequence_length + 1,
                num_iso_frames - self.sequence_length
            )

            if num_samples <= 0:
                continue  # Skip sequences that are too short

            # Create samples
            for i in range(num_samples):
                sample = {
                    'sequence_id': seq_id,
                    'fire_frame_indices': list(range(i, i + self.sequence_length)),
                    'iso_target_index': i + self.sequence_length,
                    'fire_seq_path': fire_seq_path,
                    'iso_seq_path': iso_seq_path,
                    'fire_frame_files': fire_frame_files,
                    'iso_frame_files': iso_frame_files
                }
                self.samples.append(sample)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load past fire frames and current fire frame
        past_frames_masks = []
        for frame_idx in sample['fire_frame_indices']:
            frame_file = sample['fire_frame_files'][frame_idx]
            frame_path = os.path.join(sample['fire_seq_path'], frame_file)
            frame_image = read_image(frame_path)
            frame_mask = torch.where(frame_image[1] == 231, 1.0, 0.0)
            if self.transform:
                frame_image = self.transform(frame_image)
            past_frames_masks.append(frame_mask)

        # Stack the past frames to create a tensor
        past_frames_tensor = torch.stack(past_frames_masks)  # Shape: [sequence_length, height, width]
        print('Past frames tensor shape:', past_frames_tensor.shape)
        past_frames_expanded = past_frames_tensor.unsqueeze(1)  # Shape: [sequence_length, 1, height, width]
        print('Past frames expanded shape:', past_frames_expanded.shape)

        # Load target isochrone
        iso_frame_file = sample['iso_frame_files'][sample['iso_target_index']]
        iso_frame_path = os.path.join(sample['iso_seq_path'], iso_frame_file)
        isochrone_image = read_image(iso_frame_path)
        isochrone_mask = torch.where(isochrone_image[1] == 231, 1.0, 0.0).unsqueeze(0)

        # Get landscape data
        y, y_, x, x_ = self._get_spatial_indices(sample)
        landscape_data = self.landscape_data[:, y:y_, x:x_].values  # Shape: [num_features, height, width]
        print('Landscape data shape:', landscape_data.shape)
        landscape_tensor = torch.from_numpy(landscape_data).float()
        print('Landscape tensor shape:', landscape_tensor.shape)
        landscape_repeated = landscape_tensor.unsqueeze(0).repeat(self.sequence_length, 1, 1, 1)  # Shape: [sequence_length, num_features, height, width]
        print('Landscape repeated shape:', landscape_repeated.shape)

        # Combine past frames and landscape data
        input_tensor = torch.cat((past_frames_expanded, landscape_repeated), dim=1)  # Shape: [sequence_length, 1 + num_features, height, width]
        print('Input tensor shape:', input_tensor.shape)

        # Load weather data
        weather_tensor = self._get_weather_data(sample)

        if self.transform:
            # Apply transform to each time step
            input_tensor_transformed = []
            for t in range(input_tensor.shape[0]):
                transformed = self.transform(input_tensor[t])
                input_tensor_transformed.append(transformed)
            input_tensor = torch.stack(input_tensor_transformed)
            isochrone_mask = self.transform(isochrone_mask)

        return (input_tensor, weather_tensor), isochrone_mask


    def __len__(self):
        return len(self.samples)

