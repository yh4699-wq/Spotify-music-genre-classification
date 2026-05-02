# Spotify-music-genre-classification

# Music Genre Classification Using PCA, t-SNE, KMeans, and XGBoost

This repository contains a data science capstone project for music genre classification using numerical audio features from Spotify-style track metadata. The goal is to predict a track's music genre from audio descriptors such as loudness, acousticness, speechiness, tempo, energy, instrumentalness, valence, and related features.

The project implements an end-to-end machine learning pipeline, including data cleaning, feature engineering, dimensionality reduction, visualization, clustering, supervised classification, and model evaluation.

## Project Overview

The dataset contains 50,000 tracks from 10 balanced music genres, with 5,000 tracks per genre. The main task is a supervised 10-class classification problem. In addition to classification, the project also explores the structure of the audio feature space using PCA, t-SNE, and KMeans clustering.

## Methods

The workflow includes:

- Data cleaning and preprocessing
- Handling missing or invalid values
- Encoding categorical variables such as musical key and mode
- Stratified train/test split by genre
- Feature standardization with z-score scaling
- Dimensionality reduction using PCA
- Multi-class classification using XGBoost
- Multi-class ROC-AUC evaluation
- t-SNE visualization of the reduced feature space
- KMeans clustering for exploratory analysis
- Feature importance analysis based on PCA components

## Main Results

The XGBoost classifier was trained on PCA-reduced features. The model achieved:

- Accuracy: 0.50 on a balanced 5,000-sample test set
- Macro F1-score: 0.50
- Macro-averaged ROC-AUC: 0.9043
- Micro-averaged ROC-AUC: approximately 0.913

The results show that some genres, such as Classical and Anime, are more separable using audio metadata, while closely related genres such as Rap and Hip-Hop are harder to distinguish due to overlapping acoustic characteristics.

## Files

- Main Jupyter Notebook containing the full implementation
- `requirements.txt`: Python package dependencies

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/yh4699-wq/Spotify-music-genre-classification-ml.git
cd music-genre-classification-ml
