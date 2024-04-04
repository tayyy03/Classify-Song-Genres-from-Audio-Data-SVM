# Classify Song Genres from Audio Data with SVM

This project uses a Support Vector Machine (SVM) to classify songs into genres based on their audio data.

## Code Overview

The code starts by importing the necessary libraries and creating a MultiLabelBinarizer object. This object is used to transform the 'genres' and 'genres_all' columns of the dataframe into a binary format suitable for machine learning.

```python
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
df['genres'] = df['genres'].apply(lambda x: [str(i) for i in x])
df['genres_all'] = df['genres_all'].apply(lambda x: [str(i) for i in x])
genres_encoded = mlb.fit_transform(df['genres'])
genres_all_encoded = mlb.fit_transform(df['genres_all'])
```
The code then creates a feature matrix X by horizontally stacking the numerical features with the encoded genre features.
```python
X = df[['bit_rate', 'duration', 'favorites', 'interest', 'listens', 'number']].values
X = np.hstack((X, genres_encoded, genres_all_encoded))
```
The target variable y is created by selecting only the songs that belong to the 'Rock' or 'Hip-Hop' genres and encoding 'Rock' as 1 and 'Hip-Hop' as 0.
```python
df_rock_hiphop = df[df['genre_top'].isin(['Rock', 'Hip-Hop'])]
y = np.where(df_rock_hiphop['genre_top'] == 'Rock', 1, 0)
```
Finally, the code visualizes the distribution of the 'Rock' and 'Hip-Hop' songs in the feature space.
```python
rock_mask = y == 1
hiphop_mask = y == 0
plt.scatter(X[rock_mask, 0], X[rock_mask, 1], c='blue', s=50, label='Rock', alpha=0.7)
plt.scatter(X[hiphop_mask, 0], X[hiphop_mask, 1], c='red', s=50, label='Hip-Hop', alpha=0.7)
plt.legend()
plt.title("Genre") 
plt.show()
```
To use this code, you need to have a dataframe df with the following columns: 'bit_rate', 'duration', 'favorites', 'interest', 'listens', 'number', 'genres', 'genres_all', and 'genre_top'. The 'genres' and 'genres_all' columns should contain lists of genres for each song, and the 'genre_top' column should contain the top genre for each song.

## Requirements
To run this code, you need to have the following Python libraries installed:

pandas
numpy
matplotlib
sklearn
You also need to have a dataframe df with the following columns: 'bit_rate', 'duration', 'favorites', 'interest', 'listens', 'number', 'genres', 'genres_all', and 'genre_top'. The 'genres' and 'genres_all' columns should contain lists of genres for each song, and the 'genre_top' column should contain the top genre for each song.

