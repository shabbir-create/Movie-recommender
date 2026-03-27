import pandas as pd
import ast

# Load dataset
movies = pd.read_csv("data/tmdb_5000_movies.csv")

# Function to extract names
def extract_names(text):
    try:
        data = ast.literal_eval(text)
        return " ".join([i["name"] for i in data])
    except:
        return ""

# Clean columns
movies["genres"] = movies["genres"].apply(extract_names)

# Keep only needed columns
df = movies[["title", "release_date", "genres"]].copy()

# Extract year
df["year"] = df["release_date"].str[:4]

# Dummy director (optional)
df["director"] = "Unknown"

# Save clean dataset
df.to_csv("data/movies_tmdb_clean.csv", index=False)

print("✅ Dataset ready!")