import os
import numpy as np
import matplotlib.pyplot as plt

# Replace with the actual path to your data directory
data_dir = "thesis/data_400/organized_spreads/"

fire_frames_root = os.path.join(data_dir, 'fire_frames')

frame_counts = []

# Get list of sequence directories
sequence_dirs = sorted(os.listdir(fire_frames_root))

# Iterate over each sequence directory
for seq_dir in sequence_dirs:
    seq_path = os.path.join(fire_frames_root, seq_dir)
    if not os.path.isdir(seq_path):
        continue  # Skip if not a directory

    # Get list of fire frame files in the sequence directory
    frame_files = [f for f in os.listdir(seq_path) if f.endswith('.png')]
    num_frames = len(frame_files)

    # Store the frame count
    frame_counts.append(num_frames)

    # Optionally, print the frame count for each sequence
    print(f'Sequence {seq_dir} has {num_frames} frames.')

# Convert frame_counts to a NumPy array for statistical computations
frame_counts_array = np.array(frame_counts)

# Calculate statistics
total_sequences = len(frame_counts_array)
min_frames = frame_counts_array.min()
max_frames = frame_counts_array.max()
mean_frames = frame_counts_array.mean()
median_frames = np.median(frame_counts_array)
std_frames = frame_counts_array.std()

print('\nSummary Statistics:')
print(f'Total sequences: {total_sequences}')
print(f'Minimum frames per sequence: {min_frames}')
print(f'Maximum frames per sequence: {max_frames}')
print(f'Mean frames per sequence: {mean_frames:.2f}')
print(f'Median frames per sequence: {median_frames}')
print(f'Standard deviation of frames per sequence: {std_frames:.2f}')

# Create a histogram
plt.figure(figsize=(10, 6))
plt.hist(frame_counts_array, bins=range(min_frames, max_frames + 2), edgecolor='black', align='left')
plt.title('Distribution of Total Frames per Sequence')
plt.xlabel('Number of Frames per Sequence')
plt.ylabel('Number of Sequences')
plt.xticks(range(min_frames, max_frames + 1))
plt.grid(axis='y', alpha=0.75)
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Assuming frame_counts_array contains the number of frames per sequence
sorted_frame_counts = np.sort(frame_counts_array)
cdf = np.arange(len(sorted_frame_counts)) / float(len(sorted_frame_counts))

plt.figure(figsize=(10, 6))
plt.plot(sorted_frame_counts, cdf)
plt.title('CDF of Frames per Sequence')
plt.xlabel('Number of Frames per Sequence')
plt.ylabel('Cumulative Probability')
plt.grid(True)
plt.show()

