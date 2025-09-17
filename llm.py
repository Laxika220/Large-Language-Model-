import numpy as np
import matplotlib.pyplot as plt

# 1. Define a simple sentence 
sentence = ["He", "checked", "his", "bank", "account"]

# 2. Create fake "word embeddings" for each word
# Each word is represented as a 4-dimensional vector
# In real LLMs, these vectors are learned and can be 300-4096 dimensions or more!
word_embeddings = {
    "He":       np.array([0.1, 0.2, 0.3, 0.4]),
    "checked":  np.array([0.5, 0.1, 0.0, 0.3]),
    "his":      np.array([0.1, 0.4, 0.2, 0.2]),
    "bank":     np.array([0.9, 0.3, 0.4, 0.1]),
    "account":  np.array([0.6, 0.5, 0.4, 0.3]),
}

# 3. Stack all word embeddings into a matrix 
# Each row represents one word's embedding
X = np.stack([word_embeddings[word] for word in sentence])

# 4. Compute the "attention scores" using dot products between word embeddings
# This tells us how similar each word is to every other word
# Result is a 5x5 matrix (since there are 5 words)
attention_scores = X @ X.T  # @ is matrix multiplication

# 5. Define a softmax function to normalize scores into probabilities
# Softmax makes sure all attention weights for a word add up to 1
def softmax(x):
    e_x = np.exp(x - np.max(x))  # subtract max for numerical stability
    return e_x / e_x.sum(axis=-1, keepdims=True)

# 6. Apply softmax to each row of the attention scores
# Now we have attention *probabilities* — how much each word attends to others
attention_probs = softmax(attention_scores)

# 7. Print out the attention weights in a readable format
print("Attention weights (how much each word focuses on the others):\n")
for i, word in enumerate(sentence):
    print(f"{word} attends to:")
    for j, other_word in enumerate(sentence):
        print(f"   {other_word}: {attention_probs[i, j]:.2f}")
    print()

# 8. Visualize the attention scores using a heatmap
# Darker color = higher attention
plt.imshow(attention_probs, cmap='Blues')
plt.xticks(ticks=np.arange(len(sentence)), labels=sentence)
plt.yticks(ticks=np.arange(len(sentence)), labels=sentence)
plt.colorbar(label='Attention Score')
plt.title("Self-Attention Heatmap")
plt.show()
