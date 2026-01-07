import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = r'C:\Users\n1sha\Desktop\zuco_complete_18subjects.pkl'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30
NUM_CLASSES = 500

print(f"Using device: {DEVICE}")
print(f"Dataset: {DATA_PATH}")

# ============================================================
# WORD CLASSIFICATION DATASET
# ============================================================

class WordClassificationDataset(Dataset):
    def __init__(self, samples, word2idx):
        self.X = []
        self.y = []
        self.words = []
        
        for sample in samples:
            word = sample['word']
            if word in word2idx:
                self.X.append(sample['eeg'])
                self.y.append(word2idx[word])
                self.words.append(word)
        
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.int64)
        
        print(f"  Samples: {len(self.X)}")
        print(f"  Classes: {len(set(self.y))}")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), self.y[idx]

# ============================================================
# MODEL ARCHITECTURES
# ============================================================

class CNNEncoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(105, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.cnn(x).squeeze(-1)
        return self.classifier(features), features

class ConformerEncoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(105, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=256, num_heads=8, dropout=0.3, batch_first=True
        )
        self.norm = nn.LayerNorm(256)
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.cnn(x).permute(0, 2, 1)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        features = x.mean(dim=1)
        return self.classifier(features), features

class CNNGRUEncoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(105, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.gru = nn.GRU(
            input_size=128, hidden_size=256, num_layers=2,
            batch_first=True, dropout=0.3
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.cnn(x).permute(0, 2, 1)
        _, h = self.gru(x)
        features = h[-1]
        return self.classifier(features), features

class CNNGRUConformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(105, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.gru = nn.GRU(
            input_size=256, hidden_size=256, num_layers=2,
            batch_first=True, dropout=0.3
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=256, num_heads=8, dropout=0.3, batch_first=True
        )
        self.norm = nn.LayerNorm(256)
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.cnn(x).permute(0, 2, 1)
        gru_out, _ = self.gru(x)
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        x = self.norm(gru_out + attn_out)
        features = x.mean(dim=1)
        return self.classifier(features), features

# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_model(model, train_loader, val_loader, model_name, num_epochs):
    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    print(f"{'='*70}")
    
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=False
    )
    
    history = {
        'train_loss': [], 'train_acc': [], 'train_top5': [], 'train_top10': [],
        'val_loss': [], 'val_acc': [], 'val_top5': [], 'val_top10': []
    }
    
    best_val_top10 = 0
    
    for epoch in range(num_epochs):
        # ============ TRAINING ============
        model.train()
        train_loss = 0
        train_correct = train_top5_correct = train_top10_correct = train_total = 0
        
        # Progress bar for training
        train_pbar = tqdm(train_loader, 
                         desc=f"Epoch {epoch+1}/{num_epochs} [TRAIN]",
                         ncols=120, 
                         leave=False)
        
        for eeg, labels in train_pbar:
            eeg, labels = eeg.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            logits, _ = model(eeg)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Accuracy metrics
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            _, top5_pred = logits.topk(5, dim=1)
            train_top5_correct += sum([labels[i] in top5_pred[i] for i in range(len(labels))])
            
            _, top10_pred = logits.topk(10, dim=1)
            train_top10_correct += sum([labels[i] in top10_pred[i] for i in range(len(labels))])
            
            # Update progress bar
            current_acc = 100. * train_correct / train_total
            current_top10 = 100. * train_top10_correct / train_total
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.2f}%',
                'top10': f'{current_top10:.2f}%'
            })
        
        train_acc = 100. * train_correct / train_total
        train_top5 = 100. * train_top5_correct / train_total
        train_top10 = 100. * train_top10_correct / train_total
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['train_top5'].append(train_top5)
        history['train_top10'].append(train_top10)
        
        # ============ VALIDATION ============
        model.eval()
        val_loss = 0
        val_correct = val_top5_correct = val_top10_correct = val_total = 0
        
        # Progress bar for validation
        val_pbar = tqdm(val_loader, 
                       desc=f"Epoch {epoch+1}/{num_epochs} [VAL]  ",
                       ncols=120, 
                       leave=False)
        
        with torch.no_grad():
            for eeg, labels in val_pbar:
                eeg, labels = eeg.to(DEVICE), labels.to(DEVICE)
                logits, _ = model(eeg)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                _, top5_pred = logits.topk(5, dim=1)
                val_top5_correct += sum([labels[i] in top5_pred[i] for i in range(len(labels))])
                
                _, top10_pred = logits.topk(10, dim=1)
                val_top10_correct += sum([labels[i] in top10_pred[i] for i in range(len(labels))])
                
                # Update progress bar
                current_acc = 100. * val_correct / val_total
                current_top10 = 100. * val_top10_correct / val_total
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{current_acc:.2f}%',
                    'top10': f'{current_top10:.2f}%'
                })
        
        val_acc = 100. * val_correct / val_total
        val_top5 = 100. * val_top5_correct / val_total
        val_top10 = 100. * val_top10_correct / val_total
        
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        history['val_top5'].append(val_top5)
        history['val_top10'].append(val_top10)
        
        scheduler.step(val_top10)
        
        # Print epoch summary
        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train: Loss={history['train_loss'][-1]:.4f}, "
              f"Top1={train_acc:.2f}%, Top5={train_top5:.2f}%, Top10={train_top10:.2f}% | "
              f"Val: Loss={history['val_loss'][-1]:.4f}, "
              f"Top1={val_acc:.2f}%, Top5={val_top5:.2f}%, Top10={val_top10:.2f}%", 
              end='')
        
        if val_top10 > best_val_top10:
            best_val_top10 = val_top10
            torch.save(model.state_dict(), f'{model_name}_best.pth')
            print(f" ✅ New best!")
        else:
            print()
    
    print(f"\n{'='*70}")
    print(f"✅ Best Val Top-10 Accuracy: {best_val_top10:.2f}%")
    
    if best_val_top10 > 31.04:
        improvement = best_val_top10 - 31.04
        print(f"🎉 BEATS BELT (31.04%) by {improvement:.2f}%!")
    else:
        gap = 31.04 - best_val_top10
        print(f"⚠️  Below BELT (31.04%) by {gap:.2f}%")
    print(f"{'='*70}")
    
    return history, best_val_top10

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)
    
    with open(DATA_PATH, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"✅ Total samples: {len(dataset)}")
    
    print("\n" + "="*70)
    print("ANALYZING VOCABULARY")
    print("="*70)
    
    all_words = [s['word'] for s in dataset]
    word_counts = Counter(all_words)
    
    top_words = [word for word, _ in word_counts.most_common(NUM_CLASSES)]
    word2idx = {word: idx for idx, word in enumerate(top_words)}
    
    print(f"Total unique words: {len(word_counts)}")
    print(f"Selected vocabulary: {NUM_CLASSES} most frequent words")
    print(f"Top 10 words: {word_counts.most_common(10)}")
    
    filtered_dataset = [s for s in dataset if s['word'] in word2idx]
    print(f"✅ Filtered samples: {len(filtered_dataset)}")
    
    print("\n" + "="*70)
    print("CREATING TRAIN/VAL SPLIT")
    print("="*70)
    
    train_samples, val_samples = train_test_split(
        filtered_dataset, test_size=0.15, random_state=42,
        stratify=[s['word'] for s in filtered_dataset]
    )
    
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    
    print("\n" + "="*70)
    print("CREATING DATASETS")
    print("="*70)
    
    train_dataset = WordClassificationDataset(train_samples, word2idx)
    val_dataset = WordClassificationDataset(val_samples, word2idx)
    
    # Fixed: num_workers=0 for Windows compatibility
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0, 
        pin_memory=True if DEVICE == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=0, 
        pin_memory=True if DEVICE == 'cuda' else False
    )
    
    print(f"✅ Train batches: {len(train_loader)}")
    print(f"✅ Val batches: {len(val_loader)}")
    
    # Define models
    models = {
        'CNN_Only': CNNEncoder(NUM_CLASSES),
        'Conformer': ConformerEncoder(NUM_CLASSES),
        'CNN_GRU': CNNGRUEncoder(NUM_CLASSES),
        'CNN_GRU_Conformer': CNNGRUConformer(NUM_CLASSES)
    }
    
    results = {}
    
    # Train all models
    print("\n" + "="*70)
    print("STARTING MODEL TRAINING")
    print("="*70)
    
    for idx, (name, model) in enumerate(models.items(), 1):
        print(f"\n[{idx}/{len(models)}] {name}")
        history, best_top10 = train_model(model, train_loader, val_loader, name, NUM_EPOCHS)
        results[name] = {'history': history, 'best_top10': best_top10}
    
    # Plotting
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Top-10 Validation Accuracy
    for name, data in results.items():
        axes[0, 0].plot(data['history']['val_top10'], label=name, linewidth=2)
    axes[0, 0].set_title('Top-10 Validation Accuracy (BELT Metric)', fontsize=14, fontweight='bold')
    axes[0, 0].axhline(y=31.04, color='r', linestyle='--', linewidth=2, label='BELT Baseline')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Top-10 Accuracy (%)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Top-1 Validation Accuracy
    for name, data in results.items():
        axes[0, 1].plot(data['history']['val_acc'], label=name, linewidth=2)
    axes[0, 1].set_title('Top-1 Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Top-1 Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Training Loss
    for name, data in results.items():
        axes[1, 0].plot(data['history']['train_loss'], label=name, linewidth=2)
    axes[1, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Best Top-10 Accuracy Comparison
    names = list(results.keys())
    best_accs = [results[name]['best_top10'] for name in names]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = axes[1, 1].bar(names, best_accs, color=colors)
    axes[1, 1].set_title('Best Top-10 Accuracy vs BELT (31.04%)', fontsize=14, fontweight='bold')
    axes[1, 1].axhline(y=31.04, color='r', linestyle='--', linewidth=2, label='BELT')
    axes[1, 1].set_ylabel('Top-10 Accuracy (%)')
    axes[1, 1].set_ylim([0, max(best_accs + [31.04]) * 1.2])
    
    for bar, acc in zip(bars, best_accs):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{acc:.2f}%', ha='center', va='bottom',
                       fontsize=11, fontweight='bold')
    
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].tick_params(axis='x', rotation=15)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('word_classification_results.png', dpi=150, bbox_inches='tight')
    print("✅ Saved plot: word_classification_results.png")
    plt.show()
    
    # Final Results
    print("\n" + "="*70)
    print("FINAL RESULTS - WORD CLASSIFICATION (Top 500 Words)")
    print("="*70)
    print(f"BELT Baseline: 31.04% Top-10 Accuracy")
    print("-"*70)
    print(f"{'Model':<25} {'Top-1':<10} {'Top-5':<10} {'Top-10':<10} {'vs BELT':<12}")
    print("-"*70)
    
    for name, data in results.items():
        top1 = max(data['history']['val_acc'])
        top5 = max(data['history']['val_top5'])
        top10 = data['best_top10']
        diff = top10 - 31.04
        symbol = "✅" if diff > 0 else "❌"
        print(f"{name:<25} {top1:>8.2f}% {top5:>8.2f}% {top10:>8.2f}% {symbol} {diff:>+6.2f}%")
    
    best_model = max(results.items(), key=lambda x: x[1]['best_top10'])
    print("\n" + "="*70)
    print(f"WINNER: {best_model[0]}")
    print(f"   Top-10 Accuracy: {best_model[1]['best_top10']:.2f}%")
    
    if best_model[1]['best_top10'] > 31.04:
        improvement = best_model[1]['best_top10'] - 31.04
        print(f"Baseline comparision {improvement:.2f}%!")
    
    print("="*70)


if __name__ == "__main__":
    main()