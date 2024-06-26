from EveBrain import train_intent_model, get_response, load_intents, prepare_data, NeuralNetwork




# Load intents and prepare data
intents_file = 'intents.json'
intents = load_intents(intents_file)
X, y, tokenizer, label_encoder, max_length = prepare_data(intents)


# Initialize and train the model
hidden_size1 = 128
hidden_size2 = 64
learning_rate = 0.001
epochs = 10000

model = NeuralNetwork(input_size=max_length, hidden_size1=hidden_size1, hidden_size2=hidden_size2, output_size=len(label_encoder.classes_))
model.train(X, y, epochs, learning_rate)

# Load the best model
model.load_model('EveBrain.npz')

# Get responses for new inputs
print(get_response(model, tokenizer, label_encoder, max_length, intents, "Hello today is a sad day"))
print(get_response(model, tokenizer, label_encoder, max_length, intents, "Thank you"))
print(get_response(model, tokenizer, label_encoder, max_length, intents, "Tell me a joke"))