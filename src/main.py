from models.dataset import FireDataset
transform = None
sequence_length = 6
data_dir = "thesis/data_400/organized_spreads/"

dataset = FireDataset(data_dir, sequence_length=sequence_length, transform=transform)
print(f'Total samples in dataset: {len(dataset)}')
for i in range(0, 5):
    sample_info = dataset.samples[i]
    print('Sequence', sample_info['sequence_id'])

    print('Past Fire Frames:')
    for idx in sample_info['fire_frame_indices']:
        print(sample_info['fire_frame_files'][idx])

    print('Target Isochrone:')
    print(sample_info['iso_frame_files'][sample_info['iso_target_index']])
    print('\n')
