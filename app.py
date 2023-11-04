import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from visual_search_vae import VAE, encoder, decoder
ds = tfp.distributions
DIMS = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
TRAIN_BUF = 60000
BATCH_SIZE = 512
DIMS = (28, 28, 1)
N_TRAIN_BATCHES = int(TRAIN_BUF/BATCH_SIZE)

train_images = x_train.reshape(x_train.shape[0], 28, 28, 1).astype("float32") / 255.0
train_dataset = (
    tf.data.Dataset.from_tensor_slices(train_images)
    .shuffle(TRAIN_BUF)
    .batch(BATCH_SIZE)
)

# Load the VAE model
optimizer = tf.keras.optimizers.Adam(1e-3)
loaded_model = VAE(
    enc=encoder,
    dec=decoder,
    optimizer=optimizer,
)
loaded_model.load_weights('vae_model_weights')

#Query Nearest Neighbors
embeddigns, _ = tf.split(loaded_model.enc(train_images), num_or_size_splits=2, axis=1)

# Define a function to query the model with user input
def query(image_id, k):
    query_embedding = embeddigns[image_id]
    distances = np.zeros(len(embeddigns))
    for i, e in enumerate(embeddigns):
        distances[i] = np.linalg.norm(query_embedding - e)
    return np.argpartition(distances, k)[:k]

# Streamlit UI
st.title("VAE Image Query App")

# User input for query image ID and k
query_image_id = st.number_input("Enter the Query Image ID (0-59999):", value=15, min_value=0, max_value=59999, key = 'query_image_id')
k = st.number_input("Number of Similar Images (k):", value=6, min_value=1, max_value=10, key= "k")

# Query the model and get indices of similar images
similar_indices = query(query_image_id, k=k)

# Display the query image
st.subheader("Query Image:")
st.image(1 - train_images[query_image_id, :], use_column_width=True,  width=200,caption=f"Image ID: {query_image_id}")

# Display similar images
st.subheader(f"Top {k} Similar Images:")
columns = st.columns(k)

for i in range(k):
    with columns[i]:
        st.image(1 - train_images[similar_indices[i], :],  width=200,use_column_width=True, caption=f"Image ID: {similar_indices[i]}")

# Display in Streamlit app
st.pyplot(plt)

