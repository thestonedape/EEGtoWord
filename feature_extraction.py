import h5py
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import time

def extract_eeg(f, ref):
    """Extract EEG array from MATLAB reference"""
    try:
        dataset = f[ref]
        data = dataset[()]
        
        if data.dtype == object:
            inner_ref = data.flat[0]
            if isinstance(inner_ref, h5py.h5r.Reference):
                actual_dataset = f[inner_ref]
                eeg_array = actual_dataset[:]
                return eeg_array
        return data
    except:
        return None

def extract_text(f, ref):
    """Extract text from MATLAB reference"""
    try:
        dataset = f[ref]
        data = dataset[()]
        if isinstance(data, np.ndarray):
            return ''.join(chr(int(c)) for c in data.flatten())
        return str(data)
    except:
        return "???"

def standardize_length(eeg, target_length=250):
    """Pad or crop EEG to target length"""
    channels, timesteps = eeg.shape
    
    if timesteps < target_length:
        padding = target_length - timesteps
        eeg = np.pad(eeg, ((0, 0), (0, padding)), mode='constant')
    elif timesteps > target_length:
        eeg = eeg[:, :target_length]
    
    return eeg

def extract_subject(mat_file_path, subject_id, min_timesteps=10):
    """Extract all word-level EEG data from one subject"""
    print(f"\n{'='*70}")
    print(f"Processing: {subject_id}")
    print(f"{'='*70}")
    
    all_data = []
    skipped = 0
    
    try:
        with h5py.File(mat_file_path, 'r') as f:
            sentence_data = f['sentenceData']
            word_refs = sentence_data['word']
            
            num_sentences = word_refs.shape[0]
            
            # Process each sentence
            for sent_idx in tqdm(range(num_sentences), desc=f"{subject_id}", ncols=80):
                try:
                    sent_word_ref = word_refs[sent_idx, 0]
                    sent_words = f[sent_word_ref]
                    
                    raw_eeg_refs = sent_words['rawEEG']
                    word_content_refs = sent_words['content']
                    
                    num_words = raw_eeg_refs.shape[0]
                    
                    # Extract each word
                    for word_idx in range(num_words):
                        word_eeg_ref = raw_eeg_refs[word_idx, 0]
                        word_text_ref = word_content_refs[word_idx, 0]
                        
                        eeg_array = extract_eeg(f, word_eeg_ref)
                        word_text = extract_text(f, word_text_ref)
                        
                        # Validate
                        if eeg_array is None or eeg_array.ndim != 2:
                            skipped += 1
                            continue
                        
                        # Handle orientation
                        if eeg_array.shape[1] == 105:
                            if eeg_array.shape[0] < min_timesteps:
                                skipped += 1
                                continue
                            eeg_array = eeg_array.T  # (105, timesteps)
                        elif eeg_array.shape[0] == 105:
                            if eeg_array.shape[1] < min_timesteps:
                                skipped += 1
                                continue
                        else:
                            skipped += 1
                            continue
                        
                        # Store
                        all_data.append({
                            'eeg': eeg_array.astype(np.float32),
                            'word': word_text,
                            'subject': subject_id,
                            'sentence_id': sent_idx,
                            'word_id': word_idx
                        })
                        
                except Exception as e:
                    continue
        
        print(f"✅ Extracted: {len(all_data):,} words | ⚠️ Skipped: {skipped:,}")
        return all_data
        
    except Exception as e:
        print(f"❌ ERROR processing {subject_id}: {e}")
        return []

def main():
    """Extract all subjects"""
    
    # Configuration
    DATA_DIR = r"C:\Users\n1sha\Desktop\zuco\task1 - NR\Matlab files"
    OUTPUT_FILE = r"C:\Users\n1sha\Desktop\zuco_complete_18subjects.pkl"
    TARGET_LENGTH = 250
    
    # ALL 18 SUBJECTS
    SUBJECTS = [
        'YAC', 'YAG', 'YAK', 'YDG', 'YDR',  # First 5 (you already have)
        'YFR', 'YFS', 'YHS', 'YIS', 'YLS',  # Next 5
        'YMD', 'YMS', 'YRH', 'YRK', 'YRP',  # Next 5
        'YSD', 'YSL', 'YTL'                  # Last 3
    ]
    
    print("="*70)
    print("ZUCO 2.0 COMPLETE DATA EXTRACTION - ALL 18 SUBJECTS")
    print("="*70)
    print(f"Data directory: {DATA_DIR}")
    print(f"Subjects: {len(SUBJECTS)}")
    print(f"Target length: {TARGET_LENGTH} timesteps")
    print(f"Output: {OUTPUT_FILE}")
    print("="*70)
    
    # Track overall stats
    all_data = []
    subject_stats = {}
    start_time = time.time()
    
    # Extract each subject
    for idx, subject_id in enumerate(SUBJECTS, 1):
        mat_file = Path(DATA_DIR) / f"results{subject_id}_NR.mat"
        
        if not mat_file.exists():
            print(f"\n❌ File not found: {mat_file}")
            subject_stats[subject_id] = 0
            continue
        
        print(f"\n[{idx}/{len(SUBJECTS)}] Starting {subject_id}...")
        
        # Extract
        subject_data = extract_subject(str(mat_file), subject_id)
        
        if len(subject_data) > 0:
            # Standardize lengths
            print(f"Standardizing to {TARGET_LENGTH} timesteps...")
            for sample in tqdm(subject_data, desc="Standardizing", ncols=80):
                sample['eeg'] = standardize_length(sample['eeg'], TARGET_LENGTH)
            
            all_data.extend(subject_data)
            subject_stats[subject_id] = len(subject_data)
        else:
            subject_stats[subject_id] = 0
    
    # Calculate statistics
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("EXTRACTION COMPLETE!")
    print("="*70)
    
    print(f"\n📊 SUBJECT BREAKDOWN:")
    print("-" * 70)
    for subject, count in subject_stats.items():
        bar = "█" * (count // 200)
        print(f"{subject}: {count:>6,} words  {bar}")
    
    print("\n" + "="*70)
    print("OVERALL STATISTICS:")
    print("="*70)
    print(f"Total samples: {len(all_data):,}")
    print(f"Successful subjects: {sum(1 for c in subject_stats.values() if c > 0)}/{len(SUBJECTS)}")
    
    # Word statistics
    words = [s['word'] for s in all_data]
    unique_words = set(words)
    print(f"Unique words: {len(unique_words):,}")
    print(f"Vocabulary size: {len(unique_words):,}")
    
    # Calculate approximate file size
    sample_size = all_data[0]['eeg'].nbytes
    total_size_gb = (len(all_data) * sample_size) / 1e9
    print(f"Estimated size: {total_size_gb:.2f} GB")
    print(f"Processing time: {elapsed_time/60:.1f} minutes")
    
    # Save
    print(f"\n💾 Saving dataset...")
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(all_data, f, protocol=4)
    
    actual_size_gb = Path(OUTPUT_FILE).stat().st_size / 1e9
    print(f"✅ Saved to: {OUTPUT_FILE}")
    print(f"✅ Actual file size: {actual_size_gb:.2f} GB")
    
    print("\n" + "="*70)
    print("🎉 DATASET READY FOR TRAINING!")
    print("="*70)
    print(f"✅ {len(all_data):,} samples")
    print(f"✅ {len(unique_words):,} unique words")
    print(f"✅ Shape per sample: (105, {TARGET_LENGTH})")
    print(f"✅ Ready for ML pipeline!")
    
    # Sample data
    print(f"\n📝 Sample data:")
    for i in range(min(3, len(all_data))):
        sample = all_data[i]
        print(f"  [{i+1}] Word: '{sample['word']}' | Subject: {sample['subject']} | Shape: {sample['eeg'].shape}")

if __name__ == "__main__":
    main()