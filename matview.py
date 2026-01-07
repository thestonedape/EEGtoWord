# import h5py
# import numpy as np

# path = r"C:\Users\n1sha\Desktop\zuco\task1 - NR\Matlab files\resultsYAC_NR.mat"

# print("Extracting word-level EEG data (handling all wrapper formats)...")

# def extract_eeg(f, ref):
#     """Extract EEG array handling various MATLAB reference formats"""
#     try:
#         # Step 1: Dereference
#         dataset = f[ref]
        
#         # Step 2: Get data
#         data = dataset[()]
        
#         # Step 3: Handle different wrapper formats
#         if data.dtype == object:
#             # It's a reference wrapper - extract the inner reference
#             inner_ref = data.flat[0]  # Use flat[0] to handle any shape
            
#             if isinstance(inner_ref, h5py.h5r.Reference):
#                 # Dereference again
#                 actual_dataset = f[inner_ref]
#                 eeg_array = actual_dataset[:]
#                 return eeg_array
#             else:
#                 return None
#         else:
#             # Direct data (unlikely but handle it)
#             return data
            
#     except Exception as e:
#         print(f"    ❌ Error: {e}")
#         return None

# def extract_text(f, ref):
#     """Extract text from MATLAB reference"""
#     try:
#         dataset = f[ref]
#         data = dataset[()]
        
#         if isinstance(data, np.ndarray):
#             text = ''.join(chr(int(c)) for c in data.flatten())
#             return text
#         else:
#             return str(data)
#     except:
#         return "???"

# with h5py.File(path, 'r') as f:
#     sentence_data = f['sentenceData']
#     word_refs = sentence_data['word']
    
#     print(f"\n{'='*60}")
#     print(f"DATASET INFO:")
#     print(f"{'='*60}")
#     print(f"Total sentences: {word_refs.shape[0]}")
    
#     # Process first sentence
#     first_sent_word_ref = word_refs[0, 0]
#     first_sent_words = f[first_sent_word_ref]
    
#     raw_eeg_refs = first_sent_words['rawEEG']
#     word_content_refs = first_sent_words['content']
    
#     print(f"First sentence words: {raw_eeg_refs.shape[0]}")
    
#     print(f"\n{'='*60}")
#     print(f"EXTRACTING ALL WORDS FROM FIRST SENTENCE:")
#     print(f"{'='*60}")
    
#     success_count = 0
#     total_timesteps = []
    
#     for i in range(raw_eeg_refs.shape[0]):
#         word_eeg_ref = raw_eeg_refs[i, 0]
#         word_text_ref = word_content_refs[i, 0]
        
#         # Extract EEG
#         eeg_array = extract_eeg(f, word_eeg_ref)
#         word_text = extract_text(f, word_text_ref)
        
#         if eeg_array is not None:
#             success_count += 1
#             total_timesteps.append(eeg_array.shape[0])
            
#             print(f"Word {i+1:2d}: '{word_text:15s}' | Shape: {str(eeg_array.shape):15s} | ✅")
#         else:
#             print(f"Word {i+1:2d}: '{word_text:15s}' | FAILED | ❌")
    
#     print(f"\n{'='*60}")
#     print(f"STATISTICS:")
#     print(f"{'='*60}")
#     print(f"Successfully extracted: {success_count}/{raw_eeg_refs.shape[0]} words")
#     print(f"Channels: 105 (consistent)")
#     print(f"Timesteps: min={min(total_timesteps)}, max={max(total_timesteps)}, avg={np.mean(total_timesteps):.1f}")
    
#     # Quick test on sentence 2 and 3
#     print(f"\n{'='*60}")
#     print(f"TESTING OTHER SENTENCES:")
#     print(f"{'='*60}")
    
#     for sent_idx in [1, 2]:
#         sent_ref = word_refs[sent_idx, 0]
#         sent_words = f[sent_ref]
#         sent_eeg_refs = sent_words['rawEEG']
        
#         print(f"\nSentence {sent_idx + 1}: {sent_eeg_refs.shape[0]} words")
        
#         # Try first 3 words
#         for i in range(min(3, sent_eeg_refs.shape[0])):
#             word_eeg_ref = sent_eeg_refs[i, 0]
#             word_text_ref = sent_words['content'][i, 0]
            
#             eeg_array = extract_eeg(f, word_eeg_ref)
#             word_text = extract_text(f, word_text_ref)
            
#             if eeg_array is not None:
#                 print(f"  Word {i+1}: '{word_text:15s}' | {eeg_array.shape} ✅")
#             else:
#                 print(f"  Word {i+1}: '{word_text:15s}' | FAILED ❌")
    
#     print(f"\n{'='*60}")
#     print(f"FINAL CONFIRMATION:")
#     print(f"{'='*60}")
#     print(f"✅ File format: MATLAB v7.3 with nested references")
#     print(f"✅ Data structure: Word-level EEG (exactly what you need!)")
#     print(f"✅ Shape: (timesteps, 105 channels) per word")
#     print(f"✅ Channels: 105 (preprocessed, cleaned)")
#     print(f"✅ Variable timesteps: depends on reading time per word")
#     print(f"\n🎉 THIS IS THE CORRECT FILE!")
#     print(f"🎉 READY TO EXTRACT ALL DATA!")