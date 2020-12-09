# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 19:56:34 2019

Group Recommender System

@author: Daniel Salvador Urgel
"""

import Recommender_Functions as recom
import pandas as pd
import numpy as np
import argparse
import pickle
import os

training_file='../movielens-100k/u3.base'
test_file='../movielens-100k/u3.test'
data_file='../movielens-100k/u.data'
items_file='../movielens-100k/u.item'

mf_data_folder='data_mf'

def read_data(training_file,items_file=None,num_users=None,num_items=None):
    """
        Read training and test data. Transform to user x item rating matrix.    
    
        @training_file: path to training ratings
        @test_file: path to test ratings

    """
    columns = ['user_id', 'item_id', 'rating', 'time']

    data = pd.read_csv(training_file, sep='\t', names=columns)
    
    if items_file is not None:
        items_data = pd.read_csv(items_file, sep='|',encoding='latin-1',header=None)
        items_data=items_data.rename(columns = {0:'id',1:'title',2:'release_date'})
    else:
        items_data=None
    
    if num_users is None:
        num_users = max(data.user_id.unique())
        num_items = max(data.item_id.unique())
    
    data_ratings = np.zeros((num_users, num_items))
    
    #Transform to User x Item rating matrix
    for row in data.itertuples(index=False):
        data_ratings[row.user_id - 1, row.item_id - 1] = row.rating

    return data_ratings,items_data

if __name__ == "__main__":

    group_members=[]
    debug=False
    num_recommendations=5
    size_groups=4
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", "--debug", action='store_true', help='Activate debug mode')
    parser.add_argument("-s", "--group_size", type=int, help= "Size of groups")
    parser.add_argument("-m", "--members", nargs='+', type=int, help= "List of members")
    parser.add_argument("-n", "--num_rec", type=int, help= "Number of recommendations")
    
    args = parser.parse_args()
    
    if args.debug:
        print("--- Debug mode ---")
        debug=True
        if args.group_size is not None:
                size_groups=args.group_size
    else:
        if args.members is None:
            print("You must provide a list of member ids")
            exit()
        else:
            group_members=args.members
            if args.num_rec is not None:
                num_recommendations=args.num_rec
       
    if debug:
        train_ratings, _ = read_data(training_file)
        test_ratings, _ = read_data(test_file,num_users=943,num_items=1682)
        
        #Matrix factorization
        user_factors, user_biases, item_factors, item_biases, rating_avg_dataset=recom.matrix_factorization(train_ratings)
        
        #Groups creation
        group_members, group_items = recom.generate_groups(train_ratings, test_ratings, train_ratings.shape[0], num_groups=6, size_group=size_groups)
        
        print("Groups generated (Size",size_groups,"): ")
        for group in group_members:
            print(group)
            
        #Calculate recommendations with AF method
        list_recom = recom.af(group_members, group_items,train_ratings, user_factors, user_biases, item_factors, item_biases, rating_avg_dataset)
           
        #Evaluate recommendations with test dataset
        prec_mean, rec_mean = recom.evaluate(group_members, list_recom, test_ratings)
 
        print("Mean precision: ",prec_mean)
        print("Mean recall: ",rec_mean)
        
    else:
        all_data, items_data = read_data(data_file,items_file=items_file)
        
        '''
            Check if there is data available about the matrix factorization
            If not, calculate it and save under the data folder.
        '''
        if not os.listdir(mf_data_folder) :
            user_factors, user_biases, item_factors, item_biases, rating_avg_dataset=recom.matrix_factorization(all_data)
        
            pickled_file = open(mf_data_folder+'/mf_data.pkl', 'wb')
            data = {'ufactors':user_factors, 'ubiases':user_biases,'ifactors':item_factors,'ibiases':item_biases, 'avg':rating_avg_dataset }
            pickle.dump(data, pickled_file)
        
        recommendations=recom.generate_recommendation_group([1,2,3],all_data,items_data,mf_file=mf_data_folder+'/mf_data.pkl',n_recommendations=num_recommendations)
        print(recommendations)
    