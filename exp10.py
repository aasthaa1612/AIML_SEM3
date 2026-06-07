import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv('books.csv')

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Create subplots
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

# 1. Histogram: Price distribution
axes[0,0].hist(data['price'], bins=20, alpha=0.7)
axes[0,0].set_title('Price Distribution')
axes[0,0].set_xlabel('Price')
axes[0,0].set_ylabel('Frequency')

# 2. Scatter plot: Pages vs Ratings
axes[0,1].scatter(data['pages'], data['ratings'], alpha=0.6)
axes[0,1].set_title('Pages vs Ratings')
axes[0,1].set_xlabel('Pages')
axes[0,1].set_ylabel('Ratings')

# 3. Box plot: Price by Bestseller
sns.boxplot(data=data, x='bestseller', y='price', ax=axes[0,2])
axes[0,2].set_title('Price by Bestseller')
axes[0,2].set_xlabel('Bestseller (0=No, 1=Yes)')
axes[0,2].set_ylabel('Price')

# 4. Line plot: Books added per Year
if 'year' in data.columns:
    yearly_count = data.groupby('year').size()
    axes[1,0].plot(yearly_count.index, yearly_count.values)
    axes[1,0].set_title('Books Added per Year')
    axes[1,0].set_xlabel('Year')
    axes[1,0].set_ylabel('Number of Books')
else:
    axes[1,0].axis('off')

# 5. Bar plot: Bestseller count
bestseller_count = data['bestseller'].value_counts()
axes[1,1].bar(bestseller_count.index, bestseller_count.values)
axes[1,1].set_title('Bestseller Count')
axes[1,1].set_xlabel('Bestseller (0=No, 1=Yes)')
axes[1,1].set_ylabel('Count')

# 6. Heatmap: Correlation among numeric features
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
corr_matrix = data[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, ax=axes[1,2])
axes[1,2].set_title('Correlation Heatmap')

# 7. Violin plot: Ratings distribution by Bestseller
sns.violinplot(data=data, x='bestseller', y='ratings', ax=axes[2,0])
axes[2,0].set_title('Ratings Distribution by Bestseller')
axes[2,0].set_xlabel('Bestseller (0=No, 1=Yes)')
axes[2,0].set_ylabel('Ratings')

# 8. Scatter plot: Price vs Ratings colored by Bestseller
axes[2,1].scatter(data['ratings'], data['price'], c=data['bestseller'], cmap='coolwarm', alpha=0.7)
axes[2,1].set_title('Price vs Ratings by Bestseller')
axes[2,1].set_xlabel('Ratings')
axes[2,1].set_ylabel('Price')

# 9. Pie chart: Top 6 genres
if 'genre' in data.columns:
    genre_counts = data['genre'].value_counts().head(6)
    axes[2,2].pie(genre_counts.values, labels=genre_counts.index, autopct='%1.1f%%')
    axes[2,2].set_title('Top 6 Genres')
else:
    axes[2,2].axis('off')

plt.tight_layout()
plt.show()


# Additional visualizations
plt.figure(figsize=(12, 8))

# 1. Scatter plot: Price vs Pages colored by Bestseller
plt.subplot(2, 2, 1)
sns.scatterplot(data=data, x='pages', y='price', hue='bestseller')
plt.title('Pages vs Price by Bestseller')
plt.xlabel('Pages')
plt.ylabel('Price')

# 2. Histogram: Price distribution by Bestseller
plt.subplot(2, 2, 2)
sns.histplot(data=data, x='price', hue='bestseller', kde=True)
plt.title('Price Distribution by Bestseller')
plt.xlabel('Price')
plt.ylabel('Density')

# 3. Count plot: Books per genre
plt.subplot(2, 2, 3)
if 'genre' in data.columns:
    sns.countplot(data=data, x='genre', hue='bestseller')
    plt.title('Books per Genre by Bestseller')
    plt.xlabel('Genre')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
else:
    plt.subplot(2, 2, 3).axis('off')

# 4. Hexbin equivalent: Ratings vs Price
plt.subplot(2, 2, 4)
plt.hexbin(data['ratings'], data['price'], gridsize=20, cmap='Blues')
plt.xlabel('Ratings')
plt.ylabel('Price')
plt.title('Ratings vs Price (Hexbin)')
plt.colorbar(label='Count')

plt.tight_layout()
plt.show()
