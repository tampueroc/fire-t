import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.dataset import FireDataset
from src.models.fire_transformer import FireTransformer

data_dir = ''
dataset = FireDataset(data_dir, sequence_length=6, transform=None)

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
# Get a single batch
for (input_tensor, weather_tensor), isochrone_mask in dataloader:
    print('Input tensor shape:', input_tensor.shape)
    print('Weather tensor shape:', weather_tensor.shape)
    print('Isochrone mask shape:', isochrone_mask.shape)
    break  # Only retrieve one batch for inspection

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Instantiate the model
model = FireTransformer(
    seq_length=dataset.sequence_length,  # Use the sequence length from your dataset
    in_channels=9,                       # C = Fire Spread  (1) + Number of Features (8)
    embed_dim=128,                       # Adjust as needed
    num_heads=4,                         # Adjust as needed
    num_layers=6,                        # Adjust as needed
    patch_size=16,                       # Adjust as needed
    img_size=400                         # Match the resized image size
).to(device)

# Define loss function and optimizer (same as before)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)



# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, ((input_tensor, weather_tensor), isochrone_mask) in enumerate(dataloader):
        # Move data to device
        input_tensor = input_tensor.to(device)      # [batch_size, seq_length, channels, height, width]
        weather_tensor = weather_tensor.to(device)  # [batch_size, weather_dim]
        isochrone_mask = isochrone_mask.to(device)  # [batch_size, 1, height, width]

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_tensor, weather_tensor)  # [batch_size, 1, img_size, img_size]

        # Ensure outputs match the target size
        outputs = nn.functional.interpolate(outputs, size=isochrone_mask.shape[2:], mode='bilinear', align_corners=False)

        # Compute loss
        loss = criterion(outputs, isochrone_mask.float())

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}')

    # Calculate average loss for the epoch
    epoch_loss = running_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}')
print('Training complete.')
