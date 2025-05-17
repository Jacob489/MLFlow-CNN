 # Create tiny synthetic dataset with Python
    python - <<EOF
import h5py
import numpy as np
import os

print("Creating synthetic HDF5 datasets...")

# Function to create a minimal dataset with the expected structure
def create_synthetic_dataset(filename, num_samples=10):
    with h5py.File(filename, 'w') as f:
        # Create minimal image data (5 bands, 64x64 pixels)
        f.create_dataset('image', data=np.random.rand(num_samples, 5, 64, 64).astype(np.float32))
        
        # Create redshift values
        f.create_dataset('specz_redshift', data=np.random.rand(num_samples, 1).astype(np.float32))
        
        # Create magnitude values for each band
        for band in ['g', 'r', 'i', 'z', 'y']:
            for col in ['cmodel_mag']:
                key = f'{band}_{col}'
                f.create_dataset(key, data=np.random.rand(num_samples, 1).astype(np.float32))
    
    print(f"Created {filename} with {num_samples} samples")
    return os.path.getsize(filename)

# Create train, validation and test datasets with minimal samples
train_size = create_synthetic_dataset('demo_astrodata/5x64x64_training_with_morphology.hdf5', 10)
val_size = create_synthetic_dataset('demo_astrodata/5x64x64_validation_with_morphology.hdf5', 5)
test_size = create_synthetic_dataset('demo_astrodata/5x64x64_testing_with_morphology.hdf5', 5)

print(f"Total synthetic data size: {(train_size + val_size + test_size) / 1024:.1f} KB")
EOF
    
    # Create a marker file to indicate CI mode
    touch .ci_mode
    
    echo "Synthetic dataset preparation complete!"
else
    echo "Local environment detected - using existing datasets"
fi