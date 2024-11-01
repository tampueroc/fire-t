from models.dataset import FireDataset
from torch.utils.data import DataLoader
transform = None
sequence_length = 6
data_dir = "thesis/data_400/organized_spreads/"

dataset = FireDataset(data_dir, sequence_length=sequence_length, transform=transform)
print(f'Total samples in dataset: {len(dataset)}')

# Define batch size
batch_size = 8  # Adjust based on your computational resources

# Create the DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,       # Shuffle the data for training
    num_workers=4,      # Adjust based on your system; use num_workers=0 if you encounter issues
    pin_memory=True     # Set to True if using GPU
)

for (input_tensor, weather_tensor), isochrone_mask in dataloader:
    print('Input tensor shape:', input_tensor.shape)        # Expected: [batch_size, seq_length, channels, 400, 400]
    print('Weather tensor shape:', weather_tensor.shape)    # Expected: [batch_size, 2]
    print('Isochrone mask shape:', isochrone_mask.shape)    # Expected: [batch_size, 1, 400, 400]
    break  # Only check the first batch

