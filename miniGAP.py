import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
#loading dataset
df = pd.read_csv("/storage/emulated/0/ENCFF252PLM.bed.txt", sep="\t", header=None)
df.columns = ["chr","start","end","name","score","strand","signal","p-val","q-val","peak"]
# dividing bin genome at 5000 base pairs and computing features.
bin_size = 5000
df["bin"] = df["start"] // bin_size
bin_features = df.groupby("bin")["signal"].agg(["mean","var"])
bin_features = bin_features.fillna(0)
signals = bin_features["mean"].to_numpy()
#  Generating training samples
corr_matrix = np.corrcoef(signals)
X = []
y = []
max_distance = 10  # in bins
for i in range(len(signals)):
    for j in range(i + 1, min(i + max_distance, len(signals))):

        # Features
        A = signals[i]
        B = signals[j]
        diff = abs(A - B)
        dist = j - i

        X.append([A, B, diff, dist])
        # Label rule based on correlation threshold
        label = 1 if corr_matrix[i, j] > 0.5 else 0
        y.append(label)
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)
print("Dataset size:", X.shape)
#building of tiny neural network
model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

X_t = torch.tensor(X)
y_t = torch.tensor(y).unsqueeze(1)
# training the loop
epochs= 10
for epoch in range(epochs):
    pred = model(X_t)
    loss = criterion(pred, y_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Epoch", epoch, "Loss:", loss.item())

# generation of interaction score matrix
def get_interaction_score(A, B, distance):
    with torch.no_grad():
        x = torch.tensor([[A, B, abs(A-B), distance]], dtype=torch.float32)
        return model(x).item()

# build simple matrix for first 50 bins
N = 50
score_matrix = np.zeros((N, N))

for i in range(N):
    for j in range(i+1, min(i+MAX_DISTANCE, N)):
        score = get_interaction_score(signals[i], signals[j], j-i)
        score_matrix[i, j] = score
        score_matrix[j, i] = score
# visualization (heatmap)
plt.imshow(score_matrix, cmap="hot", interpolation="nearest")
plt.title("MiniGAP Interaction Score Heatmap")
plt.colorbar()
plt.show()

print("Done.")