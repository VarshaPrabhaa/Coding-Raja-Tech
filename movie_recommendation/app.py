from flask import Flask, render_template, request
import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import KNNBasic, accuracy

app = Flask(__name__)

# Load Movie Titles data
movie_df = pd.read_csv('Movie_Titles.csv', encoding='unicode_escape')

# Load user data
user_df = pd.read_csv('u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

# Merge dataframes
df = user_df.merge(movie_df, on='item_id')

# Create user-item matrix
uii = df.pivot_table(index='user_id', columns='title', values='rating')

# Load data into Surprise format
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'title', 'rating']], reader)

# Train-test split
trainset, testset = train_test_split(data, test_size=0.2)

# Use KNNBasic algorithm
algo = KNNBasic(sim_options={'user_based': False})
algo.fit(trainset)

# Make predictions on the test set
predictions = algo.test(testset)

# Calculate RMSE
rmse = accuracy.rmse(predictions)

# Function to get top-N recommendations
def get_top_n_recommendations(predictions, n=10):
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        if uid not in top_n:
            top_n[uid] = []
        top_n[uid].append((iid, est))

    # Sort the predictions for each user and get top-N recommendations
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

# Get top-N recommendations
top_n_recommendations = get_top_n_recommendations(predictions, n=10)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['user_id'])
    recommendations = top_n_recommendations.get(user_id, [])
    return render_template('recommendations.html', user_id=user_id, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
