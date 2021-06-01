import numpy as np
import pandas as pd
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from keras.optimizers import Adam
from sklearn.decomposition import PCA
from keras.models import Sequential, Model

# READ AND LOG-TRANSFORM DATA
expr = pd.read_csv('MouseBrain_10X_1.3M.txt',sep='\t')
X = expr.values[:,0:(expr.shape[1]-1)]
Y = expr.values[:,expr.shape[1]-1]
X = np.log(X + 1)

# REDUCE DIMENSIONS WITH PRINCIPAL COMPONENT ANALYSIS (PCA)
n_input = 50
x_train = PCA(n_components = n_input).fit_transform(X); y_train = Y
plt.scatter(x_train[:, 0], x_train[:, 1], c = y_train, cmap = 'tab20', s = 10)
plt.title('Principal Component Analysis (PCA)')
plt.xlabel("PC1")
plt.ylabel("PC2")

# REDUCE DIMENSIONS WITH AUTOENCODER
model = Sequential()
model.add(Dense(30,       activation='elu', input_shape=(n_input,)))
model.add(Dense(20,       activation='elu'))
model.add(Dense(10,       activation='elu'))
model.add(Dense(2,        activation='linear', name="bottleneck"))
model.add(Dense(10,       activation='elu'))
model.add(Dense(20,       activation='elu'))
model.add(Dense(30,       activation='elu'))
model.add(Dense(n_input,  activation='sigmoid'))
model.compile(loss = 'mean_squared_error', optimizer = Adam())
model.fit(x_train, x_train, batch_size = 128, epochs = 500, verbose = 1)
encoder = Model(model.input, model.get_layer('bottleneck').output)
bottleneck_representation = encoder.predict(x_train)
plt.scatter(bottleneck_representation[:,0], bottleneck_representation[:,1], 
            c = y_train, s = 10, cmap = 'tab20')
plt.title('Autoencoder: 8 Layers')
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")


# TSNE ON PCA
model_tsne = TSNE(learning_rate = 200, n_components = 2, random_state = 123, 
                  perplexity = 90, n_iter = 1000, verbose = 1)
tsne = model_tsne.fit_transform(x_train)
plt.scatter(tsne[:, 0], tsne[:, 1], c = y_train, cmap = 'tab20', s = 10)
plt.title('tSNE on PCA')
plt.xlabel("tSNE1")
plt.ylabel("tSNE2")

# TSNE ON AUTOENCODER
model = Sequential()
model.add(Dense(10,     activation = 'elu', input_shape=(X.shape[1],)))
model.add(Dense(8,      activation = 'elu'))
model.add(Dense(6,      activation = 'elu'))
model.add(Dense(4,      activation = 'linear', name = "bottleneck"))
model.add(Dense(6,      activation = 'elu'))
model.add(Dense(8,      activation = 'elu'))
model.add(Dense(10,     activation = 'elu'))
model.add(Dense(X.shape[1],   activation = 'sigmoid'))
model.compile(loss = 'mean_squared_error', optimizer = Adam())
model.fit(X, X, batch_size = 128, epochs = 100, shuffle = True, verbose = 1)
encoder = Model(model.input, model.get_layer('bottleneck').output)
bottleneck_representation = encoder.predict(X)

model_tsne_auto = TSNE(learning_rate = 200, n_components = 2, random_state = 123, 
                       perplexity = 90, n_iter = 1000, verbose = 1)
tsne_auto = model_tsne_auto.fit_transform(bottleneck_representation)
plt.scatter(tsne_auto[:, 0], tsne_auto[:, 1], c = Y, cmap = 'tab20', s = 10)
plt.title('tSNE on Autoencoder: 8 Layers')
plt.xlabel("tSNE1")
plt.ylabel("tSNE2")

from umap import UMAP
model = UMAP(n_neighbors = 30, min_dist = 0.3, n_components = 2)
umap = model.fit_transform(X_reduced)
umap_coords = pd.DataFrame({'UMAP1':umap[:, 0], 'UMAP2':umap[:, 1]})
umap_coords.to_csv('umap_coords_10X_1.3M_MouseBrain.txt', sep='\t')
plt.scatter(umap[:, 0], umap[:, 1], c = Y, cmap = 'tab20', s = 1)
plt.title('UMAP')
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")


