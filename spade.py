import pandas as pd
from collections import defaultdict
from itertools import combinations
from google.colab import drive

drive.mount('/content/drive')

# Load CSV files into DataFrames
aisles = pd.read_csv('/content/drive/My Drive/instacart-market-basket-analysis/aisles.csv')
departments = pd.read_csv('/content/drive/My Drive/instacart-market-basket-analysis/departments.csv')
order_products_prior = pd.read_csv('/content/drive/My Drive/instacart-market-basket-analysis/order_products__prior.csv')
order_products_train = pd.read_csv('/content/drive/My Drive/instacart-market-basket-analysis/order_products__train.csv')
orders = pd.read_csv('/content/drive/My Drive/instacart-market-basket-analysis/orders.csv')
products = pd.read_csv('/content/drive/My Drive/instacart-market-basket-analysis/products.csv')

# Function to reduce memory usage
def reduce_memory_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > -128 and c_max < 127:
                    df[col] = df[col].astype('int8')
                elif c_min > -32768 and c_max < 32767:
                    df[col] = df[col].astype('int16')
                elif c_min > -2147483648 and c_max < 2147483647:
                    df[col] = df[col].astype('int32')
                else:
                    df[col] = df[col].astype('int64')
            else:
                if c_min > -3.4e38 and c_max < 3.4e38:
                    df[col] = df[col].astype('float32')
                else:
                    df[col] = df[col].astype('float64')
    return df

# Reduce memory usage
order_products_prior = reduce_memory_usage(order_products_prior)
order_products_train = reduce_memory_usage(order_products_train)
orders = reduce_memory_usage(orders)
products = reduce_memory_usage(products)

# Concatenate order_products__prior and order_products__train
order_products = pd.concat([order_products_prior, order_products_train])

# Merge order_products with orders to get user and order details
order_products = order_products.merge(orders, on='order_id', how='left')

# Merge the resulting DataFrame with products to get product details
order_products = order_products.merge(products, on='product_id', how='left')

# Merge the resulting DataFrame with aisles to get aisle details
order_products = order_products.merge(aisles, on='aisle_id', how='left')

# Merge the resulting DataFrame with departments to get department details
order_products = order_products.merge(departments, on='department_id', how='left')

# Sort and group the merged data to ensure that you process the orders in the correct sequence for each user
order_products = order_products.sort_values(by=['user_id', 'order_number'])

# Filter out users with very few transactions
min_transactions = 5
user_order_counts = order_products['user_id'].value_counts()
valid_users = user_order_counts[user_order_counts >= min_transactions].index
order_products = order_products[order_products['user_id'].isin(valid_users)]

# Filter out items that appear infrequently
min_item_frequency = 5
item_order_counts = order_products['product_id'].value_counts()
valid_items = item_order_counts[item_order_counts >= min_item_frequency].index
order_products = order_products[order_products['product_id'].isin(valid_items)]

# Ensure each transaction is unique for each user by removing duplicates
order_products = order_products.drop_duplicates(subset=['user_id', 'order_id', 'product_id'])

# Handle missing values
order_products = order_products.dropna()

# Function to generate candidate sequences
def generate_candidates(sequences, length):
    candidates = set()
    for seq in sequences:
        for item in valid_items:
            if item not in seq:  # Ensure the same item is not repeatedly added
                new_seq = seq + (item,)
                if len(new_seq) == length:
                    candidates.add(new_seq)
    return candidates

# Function to count support of each candidate sequence
def count_support(candidates, transactions):
    support_counts = defaultdict(int)
    num_candidates = len(candidates)
    for i, candidate in enumerate(candidates, 1):
        for transaction in transactions:
            if all(item in transaction for item in candidate):
                support_counts[candidate] += 1
        if i % 100 == 0 or i == num_candidates:
            print(f"Progress: {i}/{num_candidates}")
    return support_counts

# Function to prune candidates that do not meet the minimum support threshold
def prune_candidates(support_counts, min_support):
    return {seq: count for seq, count in support_counts.items() if count >= min_support}

# Function to implement the SPADE algorithm
def spade(transactions, min_support):
    sequences = {(): len(transactions)}
    length = 1
    while True:
        print(f"Generating candidates of length {length}...")
        candidates = generate_candidates(sequences.keys(), length)
        print(f"Number of candidates: {len(candidates)}")
        
        print("Counting support for candidates...")
        support_counts = count_support(candidates, transactions)
        total_support = sum(support_counts.values())
        print(f"Total support count for length {length}: {total_support}")
        
        print("Pruning candidates...")
        pruned_candidates = prune_candidates(support_counts, min_support)
        print(f"Pruned candidates: {pruned_candidates}")
        
        if not pruned_candidates:
            break
        sequences.update(pruned_candidates)
        length += 1
    return sorted(sequences.items(), key=lambda x: x[1], reverse=True)

# Prepare transactions for SPADE algorithm
transactions = order_products.groupby('user_id')['product_id'].apply(list).tolist()

# Define minimum support threshold
min_support = 0.2 * len(transactions)  # Adjusted to a lower value

# Run SPADE algorithm
frequent_patterns = spade(transactions, min_support)

# Output the frequent patterns
print(frequent_patterns)