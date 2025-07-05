# Image Captioning using CNN + LSTM

This project generates natural language captions for input images using a deep learning model combining **Convolutional Neural Networks (CNN)** and **Long Short-Term Memory (LSTM)** networks.

It uses **DenseNet201** as the feature extractor and **LSTM** as the sequence generator. The model is trained on the **Flickr8k**  dataset.

---

## Features

- CNN (DenseNet201) for visual feature extraction
- Embedding + LSTM for caption generation
- Trained on Flickr8k datasets (5 captions per image)
- Tokenizer for word indexing
- Padding and masking for variable-length sequences

## Model Architecture

1. **Image Encoder**:
   - Pretrained `DenseNet201` (ImageNet weights)
   - Extracts 1920-dim feature vector
   - Passed through a `Dense(256)` and reshaped for sequence merging

2. **Caption Decoder**:
   - Input caption sequence tokenized and embedded
   - Embeddings are merged with image features
   - LSTM processes the combined sequence
   - Output is passed through a `Dense(vocab_size, softmax)` to predict the next word

## Dataset

- **Flickr8k Dataset** (Recommended for Kaggle training)
  - 8,000 images
  - 5 captions per image
  - Available: [Kaggle Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
