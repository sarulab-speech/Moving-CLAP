# Create RIR dataset for moving sound sources
python moving_rir_generator.py --output_dir data/moving_rir_dataset --num_room_conditions_train 1000 --num_room_conditions_val 10
# Create simulated spatial sound dataset for training
python precompute_mixed_data.py

# If desired, clear RIR dataset to save space (optional)
# rm -rf data/moving_rir_dataset/train

# Consolidate precomputed features into single files for faster loading during training (optional)
# python precompute_mixed_data.py --consolidate
# rm -rf output_moving/simulated_spatial_sound/train/audio/

# Create non-spatial features for training
python create_non_spatial_precomputed.py