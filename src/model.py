import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.dataset import FireDataset
from src.models.fire_transformer import FireTransformer


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for (input_tensor, weather_tensor), isochrone_mask in val_loader:
            input_tensor = input_tensor.to(device)
            weather_tensor = weather_tensor.to(device)
            isochrone_mask = isochrone_mask.to(device)

            outputs = model(input_tensor, weather_tensor)
            outputs = nn.functional.interpolate(outputs, size=isochrone_mask.shape[2:], mode='bilinear', align_corners=False)
            loss = criterion(outputs, isochrone_mask.float())

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss


data_dir = ''
# Create datasets
transform = None
train_dataset = FireDataset(
    data_dir=data_dir,
    sequence_length=6,
    transform=transform,
    split='train'
)

val_dataset = FireDataset(
    data_dir=data_dir,
    sequence_length=6,
    transform=transform,
    split='val'
)

test_dataset = FireDataset(
    data_dir=data_dir,
    sequence_length=6,
    transform=transform,
    split='test'
)


# Define batch size
batch_size = 8  # Adjust based on your computational resources

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
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
best_val_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, ((input_tensor, weather_tensor), isochrone_mask) in enumerate(train_loader):
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

    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}] Training Loss: {epoch_loss:.4f}')
    # Validate the model
    val_loss = validate(model, val_loader, criterion, device)
    print(f'Epoch [{epoch+1}/{num_epochs}] Validation Loss: {val_loss:.4f}')

    # Save the model if it has the best validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print('Model saved.')
print('Training complete.')


def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for (input_tensor, weather_tensor), isochrone_mask in test_loader:
            input_tensor = input_tensor.to(device)
            weather_tensor = weather_tensor.to(device)
            isochrone_mask = isochrone_mask.to(device)

            outputs = model(input_tensor, weather_tensor)
            outputs = nn.functional.interpolate(outputs, size=isochrone_mask.shape[2:], mode='bilinear', align_corners=False)
            loss = criterion(outputs, isochrone_mask.float())

            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {avg_test_loss:.4f}')
    # You can add additional metrics like IoU, F1-score, etc.


# Load the best model
model.load_state_dict(torch.load('best_model.pth'))

# Evaluate on test set
test(model, test_loader, criterion, device)

