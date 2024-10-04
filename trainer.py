import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from nltk.tokenize import word_tokenize
from preprocessing import preprocess_captions
class ImageCaptioningTrainer:
    def __init__(
        self,
        encoder,
        decoder,
        train_dataset,
        val_dataset,
        vocab,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        batch_size=32,
        learning_rate=3e-4,
        max_length=20
    ):
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.device = device
        self.vocab = vocab
        self.max_length = max_length
        
        # Initialize dataloaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=2
        )
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
        self.optimizer = optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=learning_rate
        )
        
    def process_batch(self, images, annotations):
        """Process a batch of images and their annotations."""
       
        images = images.to(self.device)
        batch_input_sequences = []
        batch_target_sequences = []
        print("from process",annotations)
        # Process each image's caption in the batch
        for ann in annotations:
            # Extract caption from annotation dictionary
            print("ann",ann[""])
            caption = ann['caption']
            print("caption",caption)
            # Tokenize and convert to indices
            tokens = word_tokenize(caption.lower())
            print("tokens",tokens)
            indices = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
            indices = [self.vocab['<start>']] + indices + [self.vocab['<end>']]
            
            # Create input-target pairs
            for i in range(1, len(indices)):
                # Input sequence is everything up to current position
                input_seq = indices[:i]
                
                # Pad or truncate input sequence
                if len(input_seq) < self.max_length:
                    input_seq = input_seq + [self.vocab['<pad>']] * (self.max_length - len(input_seq))
                else:
                    input_seq = input_seq[:self.max_length]
                
                # Target is the next word
                target = indices[i]
                
                batch_input_sequences.append(input_seq)
                batch_target_sequences.append(target)
        
        # Convert to tensors
        input_sequences = torch.tensor(batch_input_sequences, dtype=torch.long).to(self.device)
        target_sequences = torch.tensor(batch_target_sequences, dtype=torch.long).to(self.device)
        
        return images, input_sequences, target_sequences
    
    def train_epoch(self):
        self.encoder.train()
        self.decoder.train()
        total_loss = 0
        
        for batch_idx, (images, annotations) in enumerate(tqdm(self.train_loader)):
            # Process the batch
#             print(annotations)
            images, input_sequences, target_sequences = self.process_batch(images, annotations)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Get image features from encoder
            image_features = self.encoder(images)
            
            # Pass through decoder
            outputs = self.decoder(input_sequences, image_features)
            
            # Calculate loss
            loss = self.criterion(outputs, target_sequences)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.encoder.eval()
        self.decoder.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, annotations in self.val_loader:
                # Process the batch
                images, input_sequences, target_sequences = self.process_batch(images, annotations)
                
                # Forward pass
                image_features = self.encoder(images)
                outputs = self.decoder(input_sequences, image_features)
                
                # Calculate loss
                loss = self.criterion(outputs, target_sequences)
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)
    
    def train(self, num_epochs):
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch()
            train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            val_losses.append(val_loss)
            
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Training Loss: {train_loss:.4f}')
            print(f'Validation Loss: {val_loss:.4f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'encoder_state_dict': self.encoder.state_dict(),
                    'decoder_state_dict': self.decoder.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'vocab': self.vocab,
                }, 'best_model.pth')
                
        return train_losses, val_losses
