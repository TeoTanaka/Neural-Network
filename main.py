import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

text = "This is a sample text for creating a language model. You can replace this with your own dataset."

chars = list(set(text))
char_to_int = {char: i for i, char in enumerate(chars)}
int_to_char = {i: char for i, char in enumerate(chars)}

seq_length = 50  # Length of each input sequence
dataX = []
dataY = []

for i in range(0, len(text) - seq_length, 1):
    seq_in = text[i:i + seq_length]
    seq_out = text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])

X = torch.tensor(dataX, dtype=torch.float32)
y = torch.tensor(dataY)

class LanguageModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LanguageModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        output, hidden = self.lstm(x.view(1, 1, -1), hidden)
        output = self.fc(output.view(1, -1))
        return output, hidden
input_size = len(chars)
hidden_size = 100
output_size = len(chars)
model = LanguageModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 1000
for epoch in range(num_epochs):
    hidden = (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size))
    loss = 0

    for i in range(X.shape[0]):
        optimizer.zero_grad()
        output, hidden = model(X[i], hidden)
        target = y[i].unsqueeze(0)
        loss += criterion(output, target)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
seed = "This is a sample text for creating a"
seed_int = [char_to_int[char] for char in seed]
generated_text = seed

with torch.no_grad():
    for i in range(200):  # Generate 200 characters
        input_seq = torch.tensor(seed_int[-seq_length:], dtype=torch.float32)
        hidden = (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size))
        output, hidden = model(input_seq, hidden)
        output_prob = nn.functional.softmax(output, dim=1).numpy()[0]
        predicted_char = np.random.choice(chars, p=output_prob)
        generated_text += predicted_char
        seed_int.append(char_to_int[predicted_char])

print(generated_text)
