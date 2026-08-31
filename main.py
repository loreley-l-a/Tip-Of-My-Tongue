from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

secret_word = "fruit"
user_guess = "apple"

embeddings = model.encode([secret_word, user_guess])
similarity = util.cos_sim(embeddings[0], embeddings[1])

print(f"Similarity score: {similarity.item():.4f}")
