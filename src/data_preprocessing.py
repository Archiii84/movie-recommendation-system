"""
Data Preprocessing Module
Downloads, cleans, and prepares the MovieLens dataset
"""

import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import config

class DownloadProgressBar(tqdm):
    """Progress bar for dataset download"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_dataset():
    """
    Downloads the MovieLens 100K dataset
    Returns: Path to extracted dataset folder
    """
    print("=" * 60)
    print("DOWNLOADING MOVIELENS 100K DATASET")
    print("=" * 60)
    
    zip_path = config.RAW_DATA_DIR / "ml-100k.zip"
    extract_path = config.RAW_DATA_DIR / config.DATASET_NAME
    
    # Check if already downloaded
    if extract_path.exists():
        print(f"✓ Dataset already exists at: {extract_path}")
        return extract_path
    
    # Download the dataset
    print(f"Downloading from: {config.DATASET_URL}")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Downloading") as t:
        urllib.request.urlretrieve(config.DATASET_URL, zip_path, reporthook=t.update_to)
    
    print(f"✓ Download complete! Size: {zip_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Extract the zip file
    print("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(config.RAW_DATA_DIR)
    
    print(f"✓ Extraction complete!")
    
    # Remove zip file to save space
    zip_path.unlink()
    print("✓ Cleaned up zip file")
    
    return extract_path

def load_ratings():
    """
    Loads the ratings data
    Returns: DataFrame with columns [user_id, movie_id, rating, timestamp]
    """
    print("\n" + "=" * 60)
    print("LOADING RATINGS DATA")
    print("=" * 60)
    
    ratings_path = config.get_rating_file_path()
    
    # Load ratings (tab-separated, no header)
    ratings = pd.read_csv(
        ratings_path,
        sep='\t',
        names=['user_id', 'movie_id', 'rating', 'timestamp'],
        engine='python'
    )
    
    print(f"✓ Loaded {len(ratings):,} ratings")
    print(f"  - Users: {ratings['user_id'].nunique():,}")
    print(f"  - Movies: {ratings['movie_id'].nunique():,}")
    print(f"  - Date range: {pd.to_datetime(ratings['timestamp'], unit='s').min().date()} to {pd.to_datetime(ratings['timestamp'], unit='s').max().date()}")
    
    return ratings

def load_movies():
    """
    Loads the movies metadata
    Returns: DataFrame with columns [movie_id, title, release_date, genres, ...]
    """
    print("\n" + "=" * 60)
    print("LOADING MOVIES DATA")
    print("=" * 60)
    
    movies_path = config.get_movies_file_path()
    
    # Genre columns (19 genres in MovieLens 100K)
    genre_columns = [
        'unknown', 'Action', 'Adventure', 'Animation', 'Children',
        'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
        'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    
    # Load movies (pipe-separated)
    movies = pd.read_csv(
        movies_path,
        sep='|',
        names=['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + genre_columns,
        encoding='latin-1',
        engine='python'
    )
    
    print(f"✓ Loaded {len(movies):,} movies")
    print(f"  - Genres available: {len(genre_columns)}")
    
    return movies, genre_columns

def load_users():
    """
    Loads user demographics
    Returns: DataFrame with user information
    """
    print("\n" + "=" * 60)
    print("LOADING USERS DATA")
    print("=" * 60)
    
    users_path = config.get_users_file_path()
    
    users = pd.read_csv(
        users_path,
        sep='|',
        names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
        engine='python'
    )
    
    print(f"✓ Loaded {len(users):,} users")
    print(f"  - Age range: {users['age'].min()}-{users['age'].max()}")
    print(f"  - Gender distribution: {dict(users['gender'].value_counts())}")
    
    return users

def clean_ratings(ratings):
    """
    Cleans the ratings data
    - Removes duplicates
    - Filters users/movies with too few ratings
    """
    print("\n" + "=" * 60)
    print("CLEANING RATINGS DATA")
    print("=" * 60)
    
    original_size = len(ratings)
    
    # Remove duplicates
    ratings = ratings.drop_duplicates(subset=['user_id', 'movie_id'])
    print(f"✓ Removed {original_size - len(ratings)} duplicate ratings")
    
    # Filter users with too few ratings
    user_counts = ratings['user_id'].value_counts()
    valid_users = user_counts[user_counts >= config.MIN_USER_RATINGS].index
    ratings = ratings[ratings['user_id'].isin(valid_users)]
    print(f"✓ Filtered users: {len(valid_users):,} users with >= {config.MIN_USER_RATINGS} ratings")
    
    # Filter movies with too few ratings
    movie_counts = ratings['movie_id'].value_counts()
    valid_movies = movie_counts[movie_counts >= config.MIN_MOVIE_RATINGS].index
    ratings = ratings[ratings['movie_id'].isin(valid_movies)]
    print(f"✓ Filtered movies: {len(valid_movies):,} movies with >= {config.MIN_MOVIE_RATINGS} ratings")
    
    print(f"✓ Final dataset: {len(ratings):,} ratings ({len(ratings)/original_size*100:.1f}% retained)")
    
    return ratings

def create_user_item_matrix(ratings):
    """
    Creates a user-item matrix (users as rows, movies as columns)
    Values are ratings, NaN for unseen movies
    """
    print("\n" + "=" * 60)
    print("CREATING USER-ITEM MATRIX")
    print("=" * 60)
    
    matrix = ratings.pivot(
        index='user_id',
        columns='movie_id',
        values='rating'
    )
    
    sparsity = 1 - (matrix.notna().sum().sum() / (matrix.shape[0] * matrix.shape[1]))
    
    print(f"✓ Matrix shape: {matrix.shape[0]:,} users × {matrix.shape[1]:,} movies")
    print(f"✓ Sparsity: {sparsity*100:.2f}% (most entries are empty)")
    print(f"✓ Total entries: {matrix.shape[0] * matrix.shape[1]:,}")
    print(f"✓ Filled entries: {matrix.notna().sum().sum():,}")
    
    return matrix

def get_basic_statistics(ratings, movies):
    """
    Computes and displays basic statistics
    """
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    
    print("\n📊 Rating Distribution:")
    print(ratings['rating'].value_counts().sort_index())
    
    print(f"\n📊 Rating Statistics:")
    print(f"  - Mean: {ratings['rating'].mean():.2f}")
    print(f"  - Median: {ratings['rating'].median():.2f}")
    print(f"  - Std Dev: {ratings['rating'].std():.2f}")
    
    print(f"\n📊 User Activity:")
    user_ratings = ratings.groupby('user_id').size()
    print(f"  - Mean ratings per user: {user_ratings.mean():.1f}")
    print(f"  - Median ratings per user: {user_ratings.median():.1f}")
    print(f"  - Most active user: {user_ratings.max()} ratings")
    
    print(f"\n📊 Movie Popularity:")
    movie_ratings = ratings.groupby('movie_id').size()
    print(f"  - Mean ratings per movie: {movie_ratings.mean():.1f}")
    print(f"  - Median ratings per movie: {movie_ratings.median():.1f}")
    print(f"  - Most rated movie: {movie_ratings.max()} ratings")

def save_processed_data(ratings, movies, users, matrix):
    """
    Saves cleaned data to processed folder
    """
    print("\n" + "=" * 60)
    print("SAVING PROCESSED DATA")
    print("=" * 60)
    
    ratings.to_csv(config.PROCESSED_DATA_DIR / 'ratings_clean.csv', index=False)
    movies.to_csv(config.PROCESSED_DATA_DIR / 'movies_clean.csv', index=False)
    users.to_csv(config.PROCESSED_DATA_DIR / 'users_clean.csv', index=False)
    matrix.to_csv(config.PROCESSED_DATA_DIR / 'user_item_matrix.csv')
    
    print("✓ Saved processed data files:")
    print(f"  - ratings_clean.csv")
    print(f"  - movies_clean.csv")
    print(f"  - users_clean.csv")
    print(f"  - user_item_matrix.csv")

def run_preprocessing_pipeline():
    """
    Main function that runs the complete preprocessing pipeline
    """
    print("\n" + "🎬" * 30)
    print("MOVIE RECOMMENDATION SYSTEM - DATA PREPROCESSING PIPELINE")
    print("🎬" * 30 + "\n")
    
    # Step 1: Download dataset
    dataset_path = download_dataset()
    
    # Step 2: Load data
    ratings = load_ratings()
    movies, genre_columns = load_movies()
    users = load_users()
    
    # Step 3: Clean data
    ratings_clean = clean_ratings(ratings)
    
    # Step 4: Create user-item matrix
    matrix = create_user_item_matrix(ratings_clean)
    
    # Step 5: Get statistics
    get_basic_statistics(ratings_clean, movies)
    
    # Step 6: Save processed data
    save_processed_data(ratings_clean, movies, users, matrix)
    
    print("\n" + "=" * 60)
    print("✅ PREPROCESSING COMPLETE!")
    print("=" * 60)
    print("Next steps:")
    print("  1. Run EDA notebook: notebooks/01_eda.ipynb")
    print("  2. Build collaborative filtering model")
    print("=" * 60 + "\n")
    
    return ratings_clean, movies, users, matrix

if __name__ == "__main__":
    # Run the preprocessing pipeline
    ratings, movies, users, matrix = run_preprocessing_pipeline()