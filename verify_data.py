import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

print("Loading dataset...")
with open(r'C:\Users\n1sha\Desktop\zuco_processed_5subjects.pkl', 'rb') as f:
    dataset = pickle.load(f)

print(f"\n{'='*60}")
print("DATASET STATISTICS")
print(f"{'='*60}")

# Basic stats
print(f"Total samples: {len(dataset)}")
print(f"Sample keys: {list(dataset[0].keys())}")
print(f"EEG shape: {dataset[0]['eeg'].shape}")
print(f"EEG dtype: {dataset[0]['eeg'].dtype}")

# Subject distribution
subjects = [s['subject'] for s in dataset]
subject_counts = Counter(subjects)
print(f"\nSamples per subject:")
for subj, count in sorted(subject_counts.items()):
    print(f"  {subj}: {count}")

# Word statistics
words = [s['word'] for s in dataset]
unique_words = set(words)
word_counts = Counter(words)

print(f"\nVocabulary:")
print(f"  Unique words: {len(unique_words)}")
print(f"  Total tokens: {len(words)}")
print(f"  Most common words: {word_counts.most_common(10)}")

# EEG statistics
all_eeg = np.array([s['eeg'] for s in dataset[:1000]])
print(f"\nEEG signal statistics (sample of 1000):")
print(f"  Mean: {all_eeg.mean():.4f}")
print(f"  Std: {all_eeg.std():.4f}")
print(f"  Min: {all_eeg.min():.4f}")
print(f"  Max: {all_eeg.max():.4f}")

# Visualize sample EEG
print(f"\n{'='*60}")
print("VISUALIZING SAMPLE EEG")
print(f"{'='*60}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Single channel over time
sample = dataset[0]
axes[0, 0].plot(sample['eeg'][0, :], label='Channel 1')
axes[0, 0].plot(sample['eeg'][50, :], label='Channel 51', alpha=0.7)
axes[0, 0].set_title(f"Word: '{sample['word']}' - Sample Channels")
axes[0, 0].set_xlabel("Timestep")
axes[0, 0].set_ylabel("Amplitude (µV)")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: All channels heatmap (FIXED)
im = axes[0, 1].imshow(sample['eeg'], aspect='auto', cmap='RdBu_r', vmin=-10, vmax=10)
axes[0, 1].set_title(f"Word: '{sample['word']}' - All Channels")
axes[0, 1].set_xlabel("Timestep")
axes[0, 1].set_ylabel("Channel")
plt.colorbar(im, ax=axes[0, 1], label='Amplitude (µV)')

# Plot 3: Different words comparison
for i, idx in enumerate([0, 100, 500, 1000]):
    if idx < len(dataset):
        s = dataset[idx]
        axes[1, 0].plot(s['eeg'][0, :], label=f"'{s['word']}'", alpha=0.7)
axes[1, 0].set_title("Different Words - Channel 1")
axes[1, 0].set_xlabel("Timestep")
axes[1, 0].set_ylabel("Amplitude (µV)")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Word length distribution
word_lengths = [len(s['word']) for s in dataset]
axes[1, 1].hist(word_lengths, bins=30, edgecolor='black', alpha=0.7)
axes[1, 1].set_title("Word Length Distribution")
axes[1, 1].set_xlabel("Characters")
axes[1, 1].set_ylabel("Count")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dataset_visualization.png', dpi=150, bbox_inches='tight')
print("✅ Saved visualization to: dataset_visualization.png")
plt.show()

print(f"\n{'='*60}")
print("DATA VERIFICATION COMPLETE!")
print(f"{'='*60}")
print("✅ Dataset is ready for training!")