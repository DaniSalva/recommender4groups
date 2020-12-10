# recommender4groups
Recommender system for groups based on collaborative filtering with matrix after-factorization.

In this method the user-item matrix is factored and the factors are calculated for each of the system users individually. Then the factors of each user belonging to a group are added to calculate the latent factors of the group and these are used to recommend new items. As an aggregation method in the implementation chosen, the weighted average of the group users factors (with the number of user evaluations as weight) was used.

## Data
The dataset used was Movielens-100k. It contains 100.000 ratings (1-5 score) from 943 users and 1682 movies.
You can download it here: https://grouplens.org/datasets/movielens/100k/

## How to run the code

#### Evaluation mode
The first mode evaluates the After Factorization method using two data files: train (u3.base) and test (u3.test). The matrix factorization is calculated with the train data and generates random groups of the size specified by command line (5 by default), on which recommendation predictions are generated. Subsequently, metrics of accuracy and recall comparing with existing recommendations in the test data.

>python RecommenderSystem_groups.py –debug –group_size 5
- debug: flag for activating evaluation mode.
- group_size: size of the random groups created for evaluation.

#### Recommender mode

The second mode generates a recommendation for a given group (_members parameter_) and optionally chosing the number of movies to return. In this case, the complete dataset has been used to perform matrix factorization (u.data) and the recommendation returns a list of movies containing both the id and title of each one of them (obtained from the u.item file).

So that the result is deterministic and the recommendations are always the same, the result of the matrix factorization is written to the _data_mf_ directory and, if they have already been calculated, in successive executions it will take this data instead of running again the gradient descent optimization algorithm and therefore generating different factors.

>python RecommenderSystem_groups.py --members 10 138 54 62 –num_rec 5
- members: user IDs from the group.
- num_rec: number of movies to recommend.




