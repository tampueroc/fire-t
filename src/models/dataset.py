import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms

class FireDataset(Dataset):
    def __init__(self, data_dir, sequence_length=3, transform=None):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.samples = []
        self._prepare_samples()

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

        # Load past fire frames from frame0 up to frame_i
        past_frames = []
        for frame_idx in sample['fire_frame_indices']:
            frame_file = sample['fire_frame_files'][frame_idx]
            frame_path = os.path.join(sample['fire_seq_path'], frame_file)
            image = read_image(frame_path).float() / 255.0  # Normalize to [0, 1]
            if self.transform:
                image = self.transform(image)
            past_frames.append(image)
        # Shape: [sequence_length_variable, channels, height, width]
        # Convert list to tensor (variable-length sequence)
        past_frames = torch.stack(past_frames)

        # Load target isochrone
        iso_frame_file = sample['iso_frame_files'][sample['iso_target_index']]
        iso_frame_path = os.path.join(sample['iso_seq_path'], iso_frame_file)
        isochrone_image = read_image(iso_frame_path).float() / 255.0  # Normalize to [0, 1]
        if self.transform:
            isochrone_image = self.transform(isochrone_image)

        return past_frames, isochrone_image


    def __len__(self):
        return len(self.samples)

