import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Example dataset loading
# Ensure your dataset columns are named as 'user_id', 'restaurant_id', and 'rating'
df = pd.read_csv('your_dataset.csv')

# Preprocessing
# Pivot the dataset to get a matrix of users and their ratings for each restaurant
pivot_table = df.pivot(index='restaurant_id', columns='user_id', values='rating').fillna(0)

# Convert the pivot table to a sparse matrix for more efficient calculations
restaurant_features = csr_matrix(pivot_table.values)

# Implementing KNN
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model_knn.fit(restaurant_features)

# Function to get recommendations
def recommend_restaurants(restaurant_id, data, model, n_recommendations):
    """
    This function returns n restaurant recommendations based on a given restaurant ID.
    """
    # Find the row of the given restaurant ID
    idx = data.index.get_loc(restaurant_id)
    # Get the nearest neighbors and their distances
    distances, indices = model.kneighbors(data.iloc[idx,:].values.reshape(1, -1), n_neighbors=n_recommendations+1)
    # Convert the indices to restaurant IDs
    raw_recommends = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
    # Get the restaurant IDs
    reverse_mapper = {v: k for k, v in data.index.to_series().items()}
    # Print recommendations
    print(f"Recommendations for Restaurant {restaurant_id}:")
    for i, (idx, dist) in enumerate(raw_recommends, 1):
        print(f"{i}: Restaurant ID = {reverse_mapper[idx]}, with similarity of {dist:.3f}")

# Example usage
# Assume we are getting recommendations for a restaurant with ID 1
recommend_restaurants(1, pivot_table, model_knn, 5)
