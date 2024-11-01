import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

def process_data(csv_path: str) -> pd.DataFrame:
    print(csv_path)
    # Dropping rows with NaN
    raw_data = pd.read_csv(csv_path, low_memory=False)
    
    # Features selected
    features = ['latitude', 'longitude', 'speed',
                'svr1', 'svr2', 'svr3', 'svr4',
                'Bitrate', 'Bitrate-RX']
    
    raw_data = raw_data[features].dropna()
    
    # two composite features
    raw_data['total_throughput'] = raw_data['Bitrate'] + raw_data['Bitrate-RX']
    raw_data['average_latency'] = raw_data[['svr1', 'svr2', 'svr3', 'svr4']].mean(axis=1)

    return raw_data


def get_clusters(df: pd.DataFrame) -> pd.DataFrame:
    kmeans = pickle.load(open("../model-development/clustering_model_kmeans.pkl", "rb"))
    scaler = pickle.load(open("../model-development/clustering_scaler_kmeans.pkl", "rb"))

    print("models loaded.")

    features = ['latitude', 'longitude', 'average_latency', 'total_throughput', 'speed']
    X = df[features]
    df['cluster'] = kmeans.predict(scaler.transform(X))
    return df


def get_quality(df: pd.DataFrame) -> pd.DataFrame:
    kmeans = pickle.load(open("../model-development/c2c_kmeans.pkl", "rb"))
    scaler = pickle.load(open("../model-development/c2c_scaler.pkl", "rb"))
    
    clusters = df[['speed', 'total_throughput', 'average_latency', 'cluster']].groupby('cluster').mean()

    X_c2c_scaled = scaler.transform(clusters[['total_throughput', 'average_latency']], clusters.index)
    
    clusters['quality'] = kmeans.predict(X_c2c_scaled)
    
    classes = clusters.groupby('class').mean()
    print(classes)
    classes['tp_latency'] = classes['total_throughput'] / classes['average_latency']
    classes.sort_values('tp_latency', inplace=True)
    classes['quality'] = range(len(classes))
    
    cluster_quality = clusters.merge(classes[['quality']], left_on='class', right_on='class')#[['quality']]
        
    df['quality'] = df.merge(cluster_quality, left_on='cluster', right_index=True)['quality']

    return df


def savefig(df: pd.DataFrame, filepath: str):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['longitude'], df['latitude'], c=df['cluster'], s=1)
    # plt.scatter(centers['longitude'], centers['latitude'], c='red', marker='X', s=20, label='Centroids')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    # plt.title(f'{k_to_plot} clusters of 5G Network Performance')
    plt.colorbar(label='Cluster')
    plt.savefig(filepath)