import json
import re
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder

def load_intents(file_path: str) -> dict:
    """Load intents from a JSON file."""
    with open(file_path, 'r') as file:
        intents = json.load(file)
    return intents

def preprocess_text(text: str) -> str:
    """Lowercase and remove non-alphanumeric characters from text."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

class Tokenizer:
    def __init__(self):
        self.word_index = {}
        self.index_word = {}
        self.vocab_size = 0

    def fit_on_texts(self, texts: list[str]):
        """Create a word index based on the input texts."""
        unique_words = set(word for text in texts for word in text.split())
        self.word_index = {word: i + 1 for i, word in enumerate(unique_words)}
        self.index_word = {i + 1: word for i, word in enumerate(unique_words)}
        self.vocab_size = len(self.word_index) + 1  # +1 for padding index 0

    def texts_to_sequences(self, texts: list[str]) -> list[list[int]]:
        """Convert texts to sequences of indices."""
        return [[self.word_index.get(word, 0) for word in text.split()] for text in texts]

    def pad_sequences(self, sequences: list[list[int]], maxlen: int) -> np.ndarray:
        """Pad sequences to the same length."""
        return np.array([seq + [0] * (maxlen - len(seq)) if len(seq) < maxlen else seq[:maxlen] for seq in sequences])

def prepare_data(intents: dict) -> tuple[np.ndarray, np.ndarray, Tokenizer, LabelEncoder, int]:
    """Prepare data for training."""
    tokenizer = Tokenizer()
    all_patterns = [preprocess_text(pattern) for intent in intents['intents'] for pattern in intent['patterns']]
    tokenizer.fit_on_texts(all_patterns)
    
    max_length = max(len(pattern.split()) for pattern in all_patterns)
    X = tokenizer.texts_to_sequences(all_patterns)
    X = tokenizer.pad_sequences(X, maxlen=max_length)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform([intent['tag'] for intent in intents['intents'] for _ in intent['patterns']])
    y = np.eye(len(label_encoder.classes_))[y]

    return X, y, tokenizer, label_encoder, max_length

def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

class NeuralNetwork:
    def __init__(self, input_size: int, hidden_size1: int, hidden_size2: int, output_size: int):
        self.W1 = np.random.randn(input_size, hidden_size1) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size1))
        self.W2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2. / hidden_size1)
        self.b2 = np.zeros((1, hidden_size2))
        self.W3 = np.random.randn(hidden_size2, output_size) * np.sqrt(2. / hidden_size2)
        self.b3 = np.zeros((1, output_size))
        self.learning_rate = 0.001
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.mW1, self.vW1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.mW2, self.vW2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.mW3, self.vW3 = np.zeros_like(self.W3), np.zeros_like(self.W3)
        self.mb1, self.vb1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.mb2, self.vb2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        self.mb3, self.vb3 = np.zeros_like(self.b3), np.zeros_like(self.b3)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the network."""
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.maximum(0, self.Z1)  # ReLU activation

        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = np.maximum(0, self.Z2)  # ReLU activation

        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = softmax(self.Z3)
        return self.A3

    def backward(self, X: np.ndarray, y: np.ndarray, output: np.ndarray):
        """Backward pass and weight update using Adam optimizer."""
        m = y.shape[0]

        dZ3 = output - y
        dW3 = np.dot(self.A2.T, dZ3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * (self.Z2 > 0)
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (self.Z1 > 0)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Adam optimizer updates
        self.mW1 = self.beta1 * self.mW1 + (1 - self.beta1) * dW1
        self.vW1 = self.beta2 * self.vW1 + (1 - self.beta2) * (dW1 ** 2)
        mW1_hat = self.mW1 / (1 - self.beta1 ** self.t)
        vW1_hat = self.vW1 / (1 - self.beta2 ** self.t)
        self.W1 -= self.learning_rate * mW1_hat / (np.sqrt(vW1_hat) + self.epsilon)

        self.mb1 = self.beta1 * self.mb1 + (1 - self.beta1) * db1
        self.vb1 = self.beta2 * self.vb1 + (1 - self.beta2) * (db1 ** 2)
        mb1_hat = self.mb1 / (1 - self.beta1 ** self.t)
        vb1_hat = self.vb1 / (1 - self.beta2 ** self.t)
        self.b1 -= self.learning_rate * mb1_hat / (np.sqrt(vb1_hat) + self.epsilon)

        self.mW2 = self.beta1 * self.mW2 + (1 - self.beta1) * dW2
        self.vW2 = self.beta2 * self.vW2 + (1 - self.beta2) * (dW2 ** 2)
        mW2_hat = self.mW2 / (1 - self.beta1 ** self.t)
        vW2_hat = self.vW2 / (1 - self.beta2 ** self.t)
        self.W2 -= self.learning_rate * mW2_hat / (np.sqrt(vW2_hat) + self.epsilon)

        self.mb2 = self.beta1 * self.mb2 + (1 - self.beta1) * db2
        self.vb2 = self.beta2 * self.vb2 + (1 - self.beta2) * (db2 ** 2)
        mb2_hat = self.mb2 / (1 - self.beta1 ** self.t)
        vb2_hat = self.vb2 / (1 - self.beta2 ** self.t)
        self.b2 -= self.learning_rate * mb2_hat / (np.sqrt(vb2_hat) + self.epsilon)

        self.mW3 = self.beta1 * self.mW3 + (1 - self.beta1) * dW3
        self.vW3 = self.beta2 * self.vW3 + (1 - self.beta2) * (dW3 ** 2)
        mW3_hat = self.mW3 / (1 - self.beta1 ** self.t)
        vW3_hat = self.vW3 / (1 - self.beta2 ** self.t)
        self.W3 -= self.learning_rate * mW3_hat / (np.sqrt(vW3_hat) + self.epsilon)

        self.mb3 = self.beta1 * self.mb3 + (1 - self.beta1) * db3
        self.vb3 = self.beta2 * self.vb3 + (1 - self.beta2) * (db3 ** 2)
        mb3_hat = self.mb3 / (1 - self.beta1 ** self.t)
        vb3_hat = self.vb3 / (1 - self.beta2 ** self.t)
        self.b3 -= self.learning_rate * mb3_hat / (np.sqrt(vb3_hat) + self.epsilon)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float):
        """Train the network."""
        self.learning_rate = learning_rate
        self.t = 0  # Timestep for Adam optimizer
        for epoch in range(epochs):
            self.t += 1
            output = self.forward(X)
            self.backward(X, y, output)
            if (epoch + 1) % 100 == 0:
                loss = -np.mean(np.log(output[np.arange(len(y)), np.argmax(y, axis=1)]))
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}')
        self.save_model('EveBrain.npz')
        
    def save_model(self, file_path: str):
        """Save model weights to a file."""
        np.savez(file_path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3)

    def load_model(self, file_path: str):
        """Load model weights from a file."""
        data = np.load(file_path)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.W3 = data['W3']
        self.b3 = data['b3']

def train_intent_model(intents_file: str, hidden_size1: int = 128, hidden_size2: int = 64, learning_rate: float = 0.001, epochs: int = 10000) -> tuple[NeuralNetwork, Tokenizer, LabelEncoder, int, dict]:
    """Train the intent classification model."""
    intents = load_intents(intents_file)
    X, y, tokenizer, label_encoder, max_length = prepare_data(intents)
    
    model = NeuralNetwork(input_size=max_length, hidden_size1=hidden_size1, hidden_size2=hidden_size2, output_size=len(label_encoder.classes_))
    model.train(X, y, epochs, learning_rate)
    
    return model, tokenizer, label_encoder, max_length, intents

def predict_intent(model: NeuralNetwork, tokenizer: Tokenizer, label_encoder: LabelEncoder, max_length: int, text: str) -> str:
    """Predict the intent of a given text."""
    processed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = tokenizer.pad_sequences(sequence, maxlen=max_length)
    output = model.forward(np.array(padded_sequence))
    intent_index = np.argmax(output)
    intent = label_encoder.inverse_transform([intent_index])[0]
    return intent

def get_response(model: NeuralNetwork, tokenizer: Tokenizer, label_encoder: LabelEncoder, max_length: int, intents: dict, text: str) -> str:
    """Get the response based on the predicted intent."""
    intent = predict_intent(model, tokenizer, label_encoder, max_length, text)
    for intent_obj in intents['intents']:
        if intent_obj['tag'] == intent:
            return random.choice(intent_obj['responses'])
