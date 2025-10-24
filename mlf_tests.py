import numpy as np
import polars as pl
import torch

from mlf_clean_copy import split_train_test, to_torch_data, RaceModel, get_random_riders


def test_random_riders(model):
    race_data = ...#TODO. init structured array
    random_race_id = 1
    nr_riders_per_race = 10
    get_random_riders(race_data, random_race_id, nr_riders_per_race)
        
def test_loss_function(model) -> None:

    ### list loss
    correct_ranking = np.array([1, 10, 30, 40])
    correct_order = np.argsort(np.argsort(correct_ranking))
    
    scores = torch.tensor([
        [99, 80, 10, 1], #insane good
        [4, 3, 2, 1], #correct
        [1, 4, 3, 2], #inbetween but first correct
        [4, 1, 3, 2], #inbetween but first incorrect
        [1, 2, 3, 4], #wrong
        [1, 10, 80, 99], #insane wrong
    ], dtype=torch.float32)

    losses = [
        model._list_preference_loss(scores[i], correct_order, correct_ranking)
        for i in range(scores.shape[0])
    ]
    print(losses)

    ### pairs
    pairs = model._get_pairs(correct_order, correct_ranking)
    print(pairs)

def main() -> None:

    #TODO: generate test data

    race_result_features = pl.read_parquet("data/features_df.parquet")
    print(race_result_features.dtypes)
    X_Y: tuple[np.ndarray, np.ndarray] = split_train_test(race_result_features)
    All = np.concatenate(X_Y)

    print(All.dtype)
    neural_net = RaceModel(All)

    torch_data = to_torch_data(All)

    test_loss_function(neural_net)

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    main()