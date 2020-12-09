# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 19:54:34 2019

Group Recommender System

@author: Daniel Salvador Urgel
"""

import numpy as np
import pickle

lambda_mf = 0.03
learning_rate_mf = 0.055
max_iterations=3
num_factors=15

def predict_user_rating(user, item,rating_avg, user_biases, item_biases,user_factors,item_factors):
    """
        Preduct ratings for a given user and a specicfic item.  
    
        @user: id of the user
        @item: id of the item
        @user_biases,item_biases: bias matrices
        @user_factors,item_factors: factor matrices
        @rating_avg: mean of ratings
        
        @return: prediction  of the rating user->item

    """
    return rating_avg + user_biases[user] + item_biases[item] + user_factors[user, :].dot(item_factors[item, :].T)

def matrix_factorization(ratings):
    
    """
        Preduct ratings for a given user and a specicfic item.  
    
        @ratings: ratings matrix user-items.
        
        @return: factor and bias matrices for users and items.

    """
    rating_row, rating_col = ratings.nonzero()
    n_ratings = len(rating_row)

    num_users = ratings.shape[0]
    num_items = ratings.shape[1]
    
    learning_rate = learning_rate_mf
    reg_term = lambda_mf

    rating_avg_dataset = np.mean(ratings[np.where(ratings != 0)])

    # initialize all unknowns with random values from -1 to 1
    user_factors = np.random.uniform(-1, 1, (ratings.shape[0], num_factors))
    item_factors = np.random.uniform(-1, 1, (ratings.shape[1], num_factors))
    
    user_biases = np.zeros(num_users)
    item_biases = np.zeros(num_items)
    
    ## Stochastic Gradient Descent algorithm to infer MF parameters.
    
    for iter in range(max_iterations):
        rating_index = np.arange(n_ratings)
        np.random.shuffle(rating_index)

        for idx in rating_index:
            user = rating_row[idx]
            item = rating_col[idx]

            pred = predict_user_rating(user, item,rating_avg_dataset, user_biases, item_biases,user_factors,item_factors)
            error = ratings[user][item] - pred

            user_factors[user] += learning_rate * ((error * item_factors[item]) - (reg_term * user_factors[user]))
            item_factors[item] += learning_rate * ((error * user_factors[user]) - (reg_term * item_factors[item]))

            user_biases[user] += learning_rate * (error - reg_term * user_biases[user])
            item_biases[item] += learning_rate * (error - reg_term * item_biases[item])
            
    return user_factors, user_biases, item_factors, item_biases, rating_avg_dataset


def non_watched_items(members, ratings): 
    """
        Calculate items which haven't been watch by any group member. 
    
        @members: list of user ids belonging to the group
        @ratings: ratings matrix (user-item)

        @return: list of items

    """
    if len(members) == 0: return []

    non_watched_items = np.argwhere(ratings[members[0]] == 0)
    for member in members:
        cur_non_eval_items = np.argwhere(ratings[member] == 0)
        non_watched_items = np.intersect1d(non_watched_items, cur_non_eval_items)
    return non_watched_items

def generate_groups(train_ratings, test_ratings, num_users, num_groups, size_group):

    """
        Generate groups for evaluation
        
        @train_ratings: train ratings matrix
        @test_ratings: test ratings matrix 
        @num_users: number of users in the database
        @num_groups: number of groups generated
        @size_groups: number of group members
        

        @return: group_members: list of groups 
                                (eg. [[group1_member1...group1_membern]...[groupn_member1...groupn_membern]])
                 items_group: list of items by group 
                              (eg. [[list items_group1] ... [list items_groupn]])

    """
    list_users = [i for i in range(num_users)]
    groups = []
    items_group = []
    min_num_movies = 50

    i=0
    while i in range(num_groups):
        group_members = np.random.choice(list_users, size = size_group, replace = False)
        candidate_items = non_watched_items(group_members, train_ratings)
        non_eval_items = non_watched_items(group_members, test_ratings)
        testable_items = np.setdiff1d(candidate_items, non_eval_items)
        if len(candidate_items) != 0 and len(testable_items) >= min_num_movies:
            groups += [group_members]
            items_group += [candidate_items]
            #delete members from complete list of users so they are not included in other group
            list_users = np.setdiff1d(list_users, group_members)
            i+=1
    return groups, items_group


def af(groups, items,ratings_matrix, user_factors, user_biases,item_factors,item_biases,ratings_avg,rating_thres=4,num_recom=30):
    """
        Implementation of the After-Factorization method.
        
        @groups: list of groups
        @items: list of items for each group
        @ratings: ratings matrix
        @user_factors, user_biases,item_factors,item_biases,ratings_avg: results of the Matrix Factorization.
        @rating_thres: rating threshold to be considered recommended
        @num_recom: number of recommendations for each group
        

        @return: Predicted recommendations for the given group and set of items sorted by importance.
        
    """    
    recommendations=[]
    item_id=0
    for group in groups:
        groupmember_factors = user_factors[group, :]
        groupmember_biases = user_biases[group]
        
        ratings_member = [np.size(ratings_matrix[member].nonzero()) for member in group]

        #aggregate the factors --> average
        group_factors = np.average(groupmember_factors, axis = 0, weights = ratings_member)
        bias = np.average(groupmember_biases, axis = 0, weights = ratings_member)

        #predict ratings for all candidate items
        group_ratings = {}
        for idx, item in enumerate(items[item_id]):
            actual_rating = ratings_avg + bias + item_biases[item] + np.dot(group_factors.T, item_factors[item])
            
            if (actual_rating > rating_thres):
                group_ratings[item] = actual_rating

        #Sort recommendations and select only the top ones
        group_ratings = sorted(group_ratings.items(), key=lambda x: x[1], reverse=True)[:num_recom]

        recommendations+=[np.array([rating_tuple[0] for rating_tuple in group_ratings])]
        item_id+=1
    return recommendations

def generate_recommendations(group_members,ratings, threshold):
    """
        Generation of recommendations with test ratings
        
        @group_members: list of groups
        @ratings: ratings matrix
        @threshold: rating threshold to be considered recommended        

        @return: Generated recommendations for the given group and movies evaluated with low ratings
        
    """
    
    unseen_items = non_watched_items(group_members, ratings)

    total_items = np.argwhere(np.logical_or(ratings[group_members[0]] >= threshold, ratings[group_members[0]] == 0)).flatten()
    dislikes = np.argwhere(np.logical_and(ratings[group_members[0]] > 0, ratings[group_members[0]] < threshold)).flatten()
    for member in group_members:
        actual_items = np.argwhere(np.logical_or(ratings[member] >= threshold, ratings[member] == 0)).flatten()
        dislikes = np.union1d(dislikes, np.argwhere(np.logical_and(ratings[member] > 0, ratings[member] < threshold)).flatten())
        total_items = np.intersect1d(total_items, actual_items)

    items = np.setdiff1d(total_items, unseen_items)
    return items, dislikes

def generate_recommendation_group(group_members, ratings,items_data, mf_file,n_recommendations=10):
    """
        Predict recommendations with AF method
        
        @group_members: group members
        @ratings: ratings matrix
        @items_data: data from movies
        @mf_file: file that contains data from MF calculation (factors, biases...)
        @n_recommendations: number of recommendations to generate

        @return: list of predicted recommendations in tuple form (ID, TITLE).
        
    """   
    #Load MF data from file
    pickled_file = open(mf_file,'rb')
    data = pickle.load(pickled_file)
     
    user_factors = data['ufactors']
    user_biases = data['ubiases']
    item_factors = data['ifactors']
    item_biases = data['ibiases']
    rating_avg_dataset = data['avg']
    
    group_members=np.array(group_members)
    group_items=non_watched_items(group_members,ratings)
    list_recom = af([group_members], [group_items], ratings,user_factors, user_biases, item_factors, item_biases, rating_avg_dataset,num_recom=n_recommendations)
    
    return [(movie,items_data[items_data.id==movie].title.values[0]) for movie in list_recom[0]]

def calculate_metrics(actual_rec, pred_rec,false_rec):
    """
        Metrics calculation
        
        @actual_rec: recommendation generated
        @pred_rec: predicted recommendation
        @false_rec: movies evaluated with low score
    
        @return: precision, recall, true positives and false positives.
        
    """    
    true_p = float(np.intersect1d(actual_rec, pred_rec).size)
    false_p = float(np.intersect1d(false_rec, pred_rec).size)
    try:
        precision = true_p / (true_p + false_p)
    except ZeroDivisionError:
        precision = np.NaN
    try:
        recall = true_p / actual_rec.size
    except ZeroDivisionError:
        recall = np.NaN
    return precision, recall, true_p, false_p

def evaluate(group_members, recommendation, test_ratings):
    """
        Evaluation of the system with different groups. It generates recommendations with test ratings and compares 
        with the predicted recommendations from the AF method.
        
        @group_members: list of groups
        @recommendation: predicted recommendations for each group
        @test_ratings: test ratings matrix

        @return: Mean of precision and recall calculated with all groups.
        
    """    
    prec_mean=0
    rec_mean=0
    for i in range(len(group_members)):
        actual_recom, false_positives = generate_recommendations(group_members[i],test_ratings,4)
        prec, rec, tp, fp = calculate_metrics(actual_recom, recommendation[i],false_positives)
        prec_mean+=prec
        rec_mean+=rec
    prec_mean=prec_mean/len(group_members)
    rec_mean=rec_mean/len(group_members)
    
    return prec_mean, rec_mean