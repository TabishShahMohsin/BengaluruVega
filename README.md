# Comprehensive ETA Prediction Pipeline

## Overview
This repository contains a suite of pipelines for predicting Expected Time of Arrival (ETA) for various urban transportation modes, including buses, auto-rickshaws, and multi-modal journeys. The project is divided into three main components, each addressing a specific prediction task.

## Table of Contents
1.  [Project 1: Bus ETA Prediction](#project-1-bus-eta-prediction)
    * [Part 1: Data Preprocessing & Segmentation](#part-1-data-preprocessing--segmentation)
    * [Part 2: Model Training & Hyperparameter Tuning](#part-2-model-training--hyperparameter-tuning)
    * [Part 3: Inference Pipeline](#part-3-inference-pipeline)
2.  [Project 2: Ride-Hailing ETA Prediction](#project-2-ride-hailing-eta-prediction)
    * [Part 1: SDR Preprocessing & Heuristic Modeling](#part-1-sdr-preprocessing--heuristic-modeling)
    * [Part 2: End-to-End Prediction Pipeline](#part-2-end-to-end-prediction-pipeline)
3.  [Project 3: Multi-Modal Journey Time Prediction (Bengaluru Last Mile)](#project-3-multi-modal-journey-time-prediction-bengaluru-last-mile)
    * [The Prediction Task](#the-prediction-task)
    * [Methodology](#methodology)
4.  [Deployment with Docker](#deployment-with-docker)
    * [General Instructions](#general-instructions)
    * [Project 1 Docker Execution](#project-1-docker-execution)
    * [Project 2 Docker Execution](#project-2-docker-execution)
    * [Project 3 Docker Execution](#project-3-docker-execution)
5.  [Core Technologies](#core-technologies)

---

## Project 1: Bus ETA Prediction

This project focuses on predicting the arrival time of buses by training a model on historical GPS ping data. The pipeline is broken down into three stages: preprocessing, training, and inference.

### Part 1: Data Preprocessing & Segmentation

This initial step transforms raw, daily, partitioned Parquet files containing vehicle GPS pings into a clean, structured dataset of stop-to-stop travel segments.

#### Workflow
1.  **Clean Reference Data**:
    * `stops_0.csv`: Removes unused columns and filters for stops that appear in defined route sequences.
    * `route_to_stop_sequence_v2.csv`: Parses string representations of stop lists into actual lists.
2.  **Process Raw GPS Data**:
    * Iterates through daily raw `.parquet` files using `polars` for efficient, lazy processing.
    * **Cleaning**: Selects necessary columns, decodes strings, converts timestamps, removes duplicates, and filters out 'CANCELED' trips.
    * **Broadcasts `route_id`**: Fills missing `route_id` values within a trip using forward/backward fill.
    * **Partitioning**: Saves cleaned daily data into separate directories partitioned by `route_id`.
3.  **Generate Travel Segments**:
    * The core `build_clean_segments` function processes pings for each trip.
    * It identifies when a vehicle enters and exits a geofenced radius (e.g., 35 meters) around each stop.
    * Calculates travel time by tracking the exit time from stop A (`tA`) and the entry time to the next stop B (`tB`).
4.  **Enrich and Filter**:
    * Segments are enriched with features like Haversine distance (`distance_m`), average speed, `start_hour`, and `day_of_week`.
    * Filters outliers with unreasonable travel times or speeds.

#### Output
* `clean_segments.parquet`: A single, clean Parquet file where each row represents a journey between two consecutive stops, ready for model training.

### Part 2: Model Training & Hyperparameter Tuning

This stage uses the cleaned segment data to train an `XGBRegressor` model, employing a rigorous hyperparameter tuning process with Optuna.

#### Methodology
1.  **Data Loading and Preparation**:
    * Loads `clean_segments.parquet`.
    * **Feature Engineering**: Uses temporal features like `start_hour` and `day_of_week`.
    * **Categorical Encoding**: Applies `LabelEncoder` to `route_id`, `from_stop`, and `to_stop`.
    * **Feature Selection**: Defines features (`X`) and the target variable (`y`, `travel_time_min`).
2.  **Hyperparameter Tuning with Optuna**:
    * An `objective` function evaluates hyperparameter sets using `TimeSeriesSplit` cross-validation (3 splits) to ensure validation occurs on future data.
    * For each trial, it trains an `XGBRegressor` and evaluates it using Root Mean Squared Error (RMSE).
    * Optuna minimizes the mean RMSE over 75 trials to find the best parameters.
3.  **Final Model Training**:
    * A final `XGBRegressor` is trained on the *entire* dataset using the best hyperparameters found by Optuna.

#### Generated Assets
* `eta_model_hypertuned.pkl`: The final, trained XGBoost model.
* `encoders_hypertuned.pkl`: The fitted `LabelEncoder` objects for categorical features.

### Part 3: Inference Pipeline

This is the final, executable pipeline for generating real-time ETA predictions from live vehicle GPS data.

#### Architecture
1.  **Load Assets**: Loads the trained model, encoders, cleaned stop data, route sequences, and pre-computed average route speeds (as a fallback).
2.  **Core Prediction Logic (`predict_future_stops`)**:
    * For a vehicle's latest GPS ping, it identifies the current route and sequence of upcoming stops.
    * It determines the "next" stop by finding the closest one in the sequence.
    * **Predicts Time to Next Stop**: Constructs a feature vector for the journey from the current location to the next stop and uses the model to predict travel time.
    * **Predicts Subsequent Segments**: Iterates through all remaining stop-to-stop segments, predicting the travel time for each.
    * **Cumulative ETA**: Accumulates predicted travel times to generate an ETA for each future stop.
3.  **Execution and I/O**:
    * Configured via `input_path` and `output_path`.
    * Accepts a `.parquet` file or a `.json` file mapping vehicles to their data paths.
    * Processes the latest ping for each trip to generate predictions.

#### Output
* A single JSON file (e.g., `output.json`) mapping route IDs to a dictionary of `stop_id: ETA_string` pairs.

---

## Project 2: Ride-Hailing ETA Prediction

This project predicts the ETA for a ride-hailing service (e.g., auto-rickshaws) by decomposing the total time into three components:
* **$P_a$ (Driver Search Time):** Time until a driver accepts the request.
* **$P_b$ (Driver-to-Pickup Time):** Time for the driver to reach the user.
* **$P_c$ (Trip Duration):** The actual ride time from pickup to destination.

### Part 1: SDR Preprocessing & Heuristic Modeling

This notebook processes raw availability data to compute a key metric: the Supply-to-Demand Ratio (SDR). This SDR is then used to define heuristic models for predicting $P_a$ and $P_b$.

#### Workflow
1.  **Load Data**: Reads and combines multiple `.parquet` files of time-slotted ride data aggregated by H3 hexagons.
2.  **Calculate SDR**: For each H3 hexagon and 15-minute time slot, it computes SDR:
    * **Supply**: Drivers ending trips or being idle.
    * **Demand**: Drivers starting trips.
    * A smoothing factor of 0.5 is added to prevent division-by-zero errors.
3.  **Aggregate & Pivot**: Averages the SDR across all source files for each `(h3_index, slot)` pair. The final data is pivoted so that rows are H3 indexes and columns are the 96 time slots.
4.  **Define Models**: Includes the Python function definitions for `pa()` and `predict_pb_exponential()`, which use the SDR to predict wait times.
5.  **Visualize**: Includes a script to plot the daily SDR pattern for a sample hexagon.

#### Output
* `processed_hex_avg_SDR.parquet`: Pivoted DataFrame containing the average SDR for every H3 hexagon and 15-minute time slot.

### Part 2: End-to-End Prediction Pipeline

This notebook implements the full pipeline to predict $P_a$, $P_b$, and $P_c$ for a list of ride requests.

#### Pipeline Steps
1.  **Configuration**: Defines key parameters for models, H3 resolution, and search radii.
2.  **Load Data**:
    * `data/input.csv`: The list of ride requests.
    * `ref_data/processed_hex_avg_SDR.parquet`: The pre-computed SDR data.
    * `ref_data/smoothed_speed_full.parquet`: Pre-computed historical average speeds for road segments.
3.  **Process Each Ride**:
    * **Weighted SDR Calculation**: For the ride's starting point, it calculates a weighted average SDR from nearby H3 hexagons within a defined radius.
    * **$P_a$ & $P_b$ Prediction**: Feeds the weighted SDR into the heuristic models (`predict_pa_robust` and `predict_pb_exponential`).
    * **$P_c$ Prediction**: Uses `osmnx` to fetch the real road network, calculates the shortest path, and estimates travel time using the historical average speeds for each road segment.
4.  **Output Results**: Saves the predicted `pa`, `pb`, and `pc` values for each ride to `out/output.json`.

---

## Project 3: Multi-Modal Journey Time Prediction (Bengaluru Last Mile)

This project contains the complete inference pipeline for the "Bengaluru Last Mile Challenge," predicting the total journey time for a multi-modal trip involving two auto-rickshaw rides and one bus journey.

### The Prediction Task

The goal is to predict the duration of 8 segments for a journey: **Origin → Auto-Rickshaw → Bus Stop → Bus → Bus Stop → Auto-Rickshaw → Destination**.

* **First Auto-Rickshaw Ride**: `a1` (Wait Time 1), `a2` (Wait Time 2), `a3` (Ride Time).
* **Bus Leg**: `a4` (Bus Wait Time), `a5` (Bus Ride Time).
* **Second Auto-Rickshaw Ride**: `a6` (Wait Time 1), `a7` (Wait Time 2), `a8` (Ride Time).

### Methodology

The pipeline uses two distinct models for the different transport modes.

#### Auto-Rickshaw Time Prediction (`a1`, `a2`, `a3`, `a6`, `a7`, `a8`)
* **Wait Time (`a1`, `a2`, `a6`, `a7`)**: Predicted using a **Supply-to-Demand Ratio (SDR)** model. Historical SDR for the pickup location's H3 grid and time-of-day is used to estimate wait times. A low SDR suggests a longer wait.
* **Ride Time (`a3`, `a8`)**: Calculated using **OSMnx** to generate a road network graph. The shortest path is found, and travel time is computed by summing the time to traverse each road segment based on historical average speeds.

#### Bus ETA Prediction (`a4`, `a5`)
* **Bus Ride Time (`a5`)**: A pre-trained model (`eta_model_hypertuned.pkl` from Project 1) predicts the travel time for individual stop-to-stop segments. The total ride time is the sum of these predictions plus a configurable buffer for intermediate stops.
* **Bus Wait Time (`a4`)**: The pipeline estimates the user's arrival time at the bus stop. It then uses live bus ping data to predict the ETA for all active buses on that route. The wait time is the difference between the user's arrival and the arrival of the first available bus.

---

## Deployment with Docker

This section provides instructions for building and running the prediction pipelines using Docker.

### General Instructions

* **Build the Image**: Navigate to the project's root directory (where the `Dockerfile` is located) and run the build command. Replace `<image-name>` with a name for your image.
    ```bash
    docker build -t <image-name> .
    ```
* **Run the Container**: Use the `docker run` command with volume mounts (`-v`) to link local directories for input and output to the directories inside the container.
    * The `--rm` flag automatically cleans up and removes the container after it finishes.
    * `-v "$(pwd)/local_input":/app/container_input`: Maps a local folder to a folder inside the container.
    * **Note for Windows Users**: Replace `$(pwd)` with `%cd%` in Command Prompt or PowerShell.

### Project 1 Docker Execution
* **Image Name**: `prediction-app`
* **Required Directory Structure**:
    ```
    /project-1-folder/
    |-- Dockerfile
    |-- requirements.txt
    |-- /codescripts
    |   |-- prediction.py
    |-- /refdata
    |   |-- (all reference data files)
    |-- /local_input_data
    |   |-- input.json
    |-- /local_output_data/
    ```
* **Build Command**:
    ```bash
    docker build -t prediction-app .
    ```
* **Run Command**:
    ```bash
    docker run --rm \
      -v "$(pwd)/local_input_data":/app/data \
      -v "$(pwd)/local_output_data":/app/out \
      prediction-app
    ```
* **Output**: `output.json` will appear in the `local_output_data` folder.

### Project 2 Docker Execution
* **Image Name**: `my-sdr-model:latest`
* **Required Directory Structure**:
    ```
    /project-2-folder/
    |-- Dockerfile
    |-- /codescripts
    |   |-- main.py
    |   |-- requirements.txt
    |-- /ref_data
    |   |-- processed_hex_avg_SDR.parquet
    |   |-- smoothed_speed_full.parquet
    |-- /input
    |   |-- input.csv
    |-- /output/
    ```
* **Build Command**:
    ```bash
    docker build -t my-sdr-model:latest .
    ```
* **Run Command**:
    ```bash
    docker run --rm \
      -v "$(pwd)/input:/app/data" \
      -v "$(pwd)/output:/app/out" \
      my-sdr-model:latest
    ```
* **Output**: `output.json` will appear in the `output` folder.

### Project 3 Docker Execution
* **Image Name**: `my-prediction-app`
* **Required Directory Structure**:
    ```
    /project-3-folder/
    |-- Dockerfile
    |-- requirements.txt
    |-- predict.py
    |-- /data
    |   |-- input.csv
    |-- /out/
    ```
* **Build Command**:
    ```bash
    docker build -t my-prediction-app .
    ```
* **Run Command**:
    ```bash
    docker run --rm \
      -v "$(pwd)/data":/app/data:ro \
      -v "$(pwd)/out":/app/out \
      my-prediction-app
    ```
* **Output**: `output.json` will appear in the `out` folder.

---

## Core Technologies
* **Data Processing & Analysis**: Pandas, Polars, NumPy
* **Machine Learning**: Scikit-learn, XGBoost
* **Hyperparameter Tuning**: Optuna
* **Geospatial Analysis**: OSMnx, NetworkX, H3, Shapely, Pyproj
* **Visualization**: Matplotlib
* **Containerization**: Docker
