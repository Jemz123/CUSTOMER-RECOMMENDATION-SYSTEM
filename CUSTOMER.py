import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Prepare the Data
# Sample user-item interaction data
data = {
    'User': ['Alice', 'Alice', 'Alice', 'Bob', 'Bob', 'Charlie', 'Charlie', 'Charlie', 'Charlie'],
    'Item': ['Item1', 'Item2', 'Item3', 'Item1', 'Item3', 'Item1', 'Item2', 'Item3', 'Item4'],
    'Rating': [5, 3, 2, 4, 5, 1, 2, 5, 3]
}

df = pd.DataFrame(data)

# Display the data
print("User-Item Interaction Data:")
print(df)

# Step 2: Build User-Item Matrix
user_item_matrix = df.pivot(index='User', columns='Item', values='Rating').fillna(0)

# Display the matrix
print("\nUser-Item Matrix:")
print(user_item_matrix)

# Step 3: Compute Similarity
# Compute cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Display user similarity matrix
print("\nUser Similarity Matrix:")
print(user_similarity_df)

# Step 4: Generate Recommendations
def recommend_items(user, user_item_matrix, user_similarity_df, top_n=3):
    # Get similar users
    similar_users = user_similarity_df[user].sort_values(ascending=False)
    similar_users = similar_users[similar_users.index != user]  # Exclude self
    
    # Get items rated by similar users
    similar_users_items = user_item_matrix.loc[similar_users.index]
    
    # Calculate weighted ratings for items
    weighted_ratings = similar_users_items.T.dot(similar_users)
    sum_of_similarities = similar_users.sum()
    recommendation_scores = weighted_ratings / sum_of_similarities
    
    # Remove items already rated by the user
    user_rated_items = user_item_matrix.loc[user] > 0
    recommendation_scores = recommendation_scores[~user_rated_items]
    
    # Get top N recommendations
    recommendations = recommendation_scores.sort_values(ascending=False).head(top_n)
    
    return recommendations

# Example: Recommend items for 'Alice'
user = 'Alice'
recommendations = recommend_items(user, user_item_matrix, user_similarity_df)
print(f"\nTop recommendations for {user}:")
print(recommendations)
