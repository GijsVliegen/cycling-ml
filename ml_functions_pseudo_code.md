# Machine Learning Functions Pseudocode

This document provides pseudocode descriptions for all functions and classes in `machine_learning_functions.py`.

## Class: NeuralNet

### __init__(init_func, input_size=1, hidden_size=10, output_size=1, lr=0.01)
- Initialize a neural network with one hidden layer
- Set up layers: Linear(input_size, hidden_size) -> ReLU -> Linear(hidden_size, output_size)
- Initialize Adam optimizer with given learning rate
- Set up MarginRankingLoss criterion

### forward(x)
- Convert input to torch tensor
- Pass through the neural network layers
- Return numpy array of predictions

### preference_update(input1, input2, higher_is_1=True)
- Compute scores for both inputs using forward pass
- Create target tensor based on higher_is_1 flag
- Compute loss using MarginRankingLoss
- Perform backward pass and optimizer step
- Return the loss value

### grad_descent_pass(predicted_neighbor_scores, true_scores, margin=1.0)
- Initialize total loss to 0 and pair count to 0
- For each pair of predictions (i, j):
  - If true_scores[i] < true_scores[j] (i should rank higher):
    - Compute loss where predicted_scores[i] should be > predicted_scores[j]
  - Else if true_scores[j] < true_scores[i] (j should rank higher):
    - Compute loss where predicted_scores[j] should be > predicted_scores[i]
  - Add loss to total and increment pair count
- Average total loss by number of pairs
- Perform backward pass and optimizer step
- Return the average loss

## Class: Y
 contains rankings, scores, orders and buckets
### __init__(ranking=None, scores=None, order=[])
- If both ranking and scores are None, raise error
- Store ranking and scores
- Compute relative order

### ranks_to_buckets()
- For each rank in ranking:
  - Find which bucket range it falls into (1-3, 4-8, 9-15, 16-25, 26+)
- Assert length matches ranking length
- Return numpy array of bucket indices

### complete_pred_with_true(Y_true=None)
- complete ranking of prediction scores from true ranking

### from_ranking(ranking)
- Create Y instance with given ranking, no scores

### from_scores(scores)
- Create Y instance with no ranking, given scores

### calculate_errors(Y_true)
- Compute errors as difference between true and predicted buckets

## Class: SplinesSGD

### __init__(X)
- Initialize feature functions as NeuralNet instances for each feature

### grad_descend_pass(errors, Y_pred, Y_true, all_neighbor_feature_scores_3d)
- For each feature function:
  - Extract neighbor scores for that feature
  - Get true estimated scores for neighbors
  - For each rider's neighbors and error:
    - Compute average true estimate for the feature
    - For each neighbor score, call preference_update with appropriate higher_is_1 flag

### get_closest_points(X, y, k)
- Find all historical points for same rider (same name, different race, same or earlier year)
- Sort by rank (ascending, lower is better)
- Return top k indices

### compute_rider_scores_constituents(all_data, rider, k=25)
- Get k closest neighbor indices
- Get neighbor feature values
- For each feature function, compute transformed scores for neighbors
- Return stacked array of feature scores

### predict_ranking_for_race(indices, data)
- Initialize lists for predictions, neighbor scores, indices, distances
- For each rider index:
  - Get neighbor indices and data
  - Compute distance-weighted differences (currently commented out)
  - Compute feature scores for neighbors
  - Compute rider score as sum of neighbor scores
  - Store results
- Stack neighbor scores into 3D array
- Create Y prediction from scores
- Return prediction, neighbor scores, indices, distances

### training_step(Y_true_ranking, indices, data)
- Create Y_true from ranking
- Predict for the race
- Compute bucket errors
- Perform gradient descent pass
- Return errors

### plot_learned_splines()
- For each feature function (spline), call plot_learned_spline()

## Class: MLflowWrapper

### __init__(model)
- Store the SplinesSGD model instance

### predict(context, race_id)
- Load full dataset from numpy file
- Get rider indices for the race
- If no riders, return empty results
- Predict rankings for the race
- Get top 25 riders by predicted ranking
- Return dictionary with top riders and scores

## Function: split_train_test(All, test_ratio=0.2)
- Compute split index as (1 - test_ratio) * total rows
- Return train array (first part) and test array (second part)

## Function: get_random_riders(All, race_id, min_nr=6)
- Get indices of top 25 and bottom performers for the race
- Take minimum of min_nr/2 from each group
- Return concatenated indices

## Function: train_model(All, X, spline_model)
- Set number of riders per race and epochs
- For each epoch:
  - Select random race from training data
  - Get random rider indices for that race
  - If not enough riders, skip
  - Get true rankings
  - Perform training step
  - Accumulate loss
  - Print loss every 10% of epochs

## Function: compute_model_performance(All, Y, model)
- Select random test races
- For each test race:
  - Get random rider indices
  - If not enough, skip
  - Predict rankings
  - Compute bucket errors
  - Collect true and predicted buckets
- Compute MSE, MAE, R2 on collected buckets
- Print distance weights and plot splines
- Return metrics dictionary

## Function: main()
- Load parquet data
- Split into train/test
- Concatenate back to full dataset
- Create SplinesSGD model
- Start MLflow run
- Log parameters
- Train model
- Compute performance
- Log metrics and model
- Print performance

## Function: predict_top25_for_race(race_id)
- Load model from MLflow
- Call predict method with race_id
- Return results

## Function: main2()
- Load parquet data
- Select random race ID
- Predict top 25 for that race