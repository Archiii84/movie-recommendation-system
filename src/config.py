
"""
Configuration file for the Movie Recommendation System
Contains all constants, paths, and hyperparameters
"""

import os
from pathlib import Path

# ==================== PROJECT PATHS ====================
# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models" / "saved_models"

# Create directories if they don't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ==================== DATASET ====================
# MovieLens 100K dataset
DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATASET_NAME = "ml-100k"

# File names
RATINGS_FILE = "u.data"
MOVIES_FILE = "u.item"
USERS_FILE = "u.user"

# ==================== DATA PARAMETERS ====================
# Rating scale
MIN_RATING = 1
MAX_RATING = 5

# Train-test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Minimum ratings threshold
MIN_USER_RATINGS = 20  # Users with less than this many ratings are filtered
MIN_MOVIE_RATINGS = 10  # Movies with less than this many ratings are filtered

# ==================== MODEL HYPERPARAMETERS ====================
# Collaborative Filtering
N_NEIGHBORS = 50  # Number of similar users/items to consider
SIMILARITY_METRIC = "cosine"  # Options: cosine, pearson, euclidean

# Hybrid Model
COLLABORATIVE_WEIGHT = 0.7  # Weight for collaborative filtering in hybrid
CONTENT_WEIGHT = 0.3  # Weight for content-based filtering in hybrid

# Recommendations
TOP_N_RECOMMENDATIONS = 10  # Default number of recommendations to return

# ==================== EVALUATION METRICS ====================
# Metrics to compute
METRICS = ["rmse", "mae", "precision", "recall"]
K_VALUES = [5, 10, 20]  # For Precision@K and Recall@K

# ==================== FEATURE ENGINEERING ====================
# Genre encoding
USE_GENRE_FEATURES = True
GENRE_WEIGHT = 1.0

# Popularity features
USE_POPULARITY = True
POPULARITY_WEIGHT = 0.5

# ==================== VISUALIZATION ====================
# Plot settings
FIGURE_SIZE = (12, 6)
STYLE = "seaborn-v0_8"
COLOR_PALETTE = "husl"

# ==================== STREAMLIT APP ====================
APP_TITLE = "🎬 Movie Recommendation System"
APP_ICON = "🎬"
PAGE_LAYOUT = "wide"

# ==================== LOGGING ====================
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ==================== REPRODUCIBILITY ====================
# Set random seeds for reproducibility
import random
import numpy as np

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ==================== HELPER FUNCTIONS ====================
def get_rating_file_path():
    """Returns the full path to the ratings file"""
    return RAW_DATA_DIR / DATASET_NAME / RATINGS_FILE

def get_movies_file_path():
    """Returns the full path to the movies file"""
    return RAW_DATA_DIR / DATASET_NAME / MOVIES_FILE

def get_users_file_path():
    """Returns the full path to the users file"""
    return RAW_DATA_DIR / DATASET_NAME / USERS_FILE

def get_model_save_path(model_name):
    """Returns the path for saving a model"""
    return MODELS_DIR / f"{model_name}.pkl"

# Print configuration on import (optional - for debugging)
if __name__ == "__main__":
    print("=" * 50)
    print("Movie Recommendation System Configuration")
    print("=" * 50)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Random State: {RANDOM_STATE}")
    print(f"Test Size: {TEST_SIZE}")
    print(f"Top N Recommendations: {TOP_N_RECOMMENDATIONS}")
    print("=" * 50)