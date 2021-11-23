import pandas as pd
import numpy as np
movies = pd.read_csv("data/movies.csv")
movies = movies.set_index("movieId")
movies_images = pd.read_csv("data/movies_with_images.csv")
ratings = pd.read_csv("data/ratings.csv")
ratings = ratings.drop("timestamp",axis=1)
movies['mean_rating'] = ratings.groupby("movieId").mean()['rating']
movies['total_votes'] = ratings["movieId"].value_counts()
movies = movies.dropna() #Drop movies that don't appear in the ratings dataframe
last_user = ratings['userId'].max()


class BestRatedRecommender:
    
    def __init__(self, user_ratings):
        best_rated = movies[(movies["total_votes"]>50)].sort_values("mean_rating",ascending=False)
        best_rated = best_rated[~best_rated.index.isin(user_ratings.index)]
        self.recommendations = best_rated
        
    def recommend(self,n=10):
        print("Here are the top {} best rated movies by our community you may have not seen yet:".format(n))
        return self.recommendations.drop(['genres','mean_rating','total_votes'],axis=1).head(n)

class RarePearlsRecommender:

    def __init__(self, user_ratings):
        best_rated = movies[(movies["total_votes"]<=50)&(movies["total_votes"]>=10)].sort_values("mean_rating",ascending=False)
        best_rated = best_rated[~best_rated.index.isin(user_ratings.index)]
        self.recommendations = best_rated
        
    def recommend(self,n=10):
        print(f"{n} less known movies our community loves:")
        return  self.recommendations.drop(['genres','mean_rating','total_votes'],axis=1).head(n)


class bubble_rec:
    
    def __init__(self, user, k = 50, n = None):
        self.n = n
        self.k = k
        self.user = user
        self.user_ratings = self.get_user_ratings(user)
        self.all_users = ratings['userId'].unique()
        if n:
            self.all_users = np.random.choice(self.all_users,size=n) #n allows the selection of less users in case of a big DataFrame
        
        # Calculates knn
        self.knn_df = self.modified_knn(user)
        
        #Get ratings from similar users
        self.similar_users = self.knn_df.index
        self.similar_users_ratings = ratings.set_index("userId").loc[self.similar_users]
        self.__recommendations = self.similar_users_ratings.groupby("movieId").mean()[['rating']]
        self.instances = self.similar_users_ratings.groupby("movieId").count()[['rating']]
        
        #Prepares recommendations matrix and calculates quantiles
        self.__recommendations = self.__recommendations.join(self.instances, lsuffix="_l", rsuffix="_r")
        self.__recommendations.columns = ['neighbors_ratings','neighbors_count']
        self.__recommendations = self.__recommendations.query("neighbors_count > %.2f" % 5)
        self.quantile90 = self.__recommendations.neighbors_count.quantile(q=0.9)
        self.quantile50 = self.__recommendations.neighbors_count.quantile(q=0.5)
        
        # Prepares border recommendations
        self.__border_recommendations = self.__recommendations.query("neighbors_count < %.2f" % (self.quantile90))
        self.__border_recommendations = self.__border_recommendations.query("neighbors_count >= %.2f" % (self.quantile50))        
        self.__border_recommendations = self.__border_recommendations.sort_values("neighbors_ratings", ascending=False)
        self.__border_recommendations = self.__border_recommendations.drop(self.user.umovieids,errors='ignore')
        self.__border_recommendations = self.__border_recommendations.join(movies) 
        
        # Finalizes recommendation matrix
        self.__recommendations = self.__recommendations.query("neighbors_count >= %.2f" % self.quantile90)  
        self.__recommendations = self.__recommendations.sort_values("neighbors_ratings", ascending=False)
        self.__recommendations = self.__recommendations.drop(user.umovieids,errors='ignore')
        self.__recommendations = self.__recommendations.join(movies)       
        
        
    def get_user_ratings(self,user): # Get ratings from user
        return pd.DataFrame(data={'movieId':user.umovieids,'rating':user.uratings}).set_index("movieId")
         
    def get_vector_norm(self,a,b): #Calculates vector norm
        return np.linalg.norm(a - b)
    
    def get_user_distance(self, user, userId2, minimum = 10): #Calculates the distance between two users
        user2_ratings = ratings[(ratings["userId"] == userId2)][['movieId','rating']].set_index("movieId")
        ratings_diff = self.user_ratings.join(user2_ratings, lsuffix="_l", rsuffix="_r").dropna()
        if(len(ratings_diff) < minimum): # If there are not enough movies in common between the pair, return None or big number.
            return None #[user.userId, userId2, 100000] 
        distance = self.get_vector_norm(ratings_diff['rating_l'],ratings_diff['rating_r'])
        distance = distance/len(ratings_diff)
        return [user.userId, userId2, distance]  
    
    def modified_knn(self,user): #KNN method
        distances = [self.get_user_distance(user, userId2) for userId2 in self.all_users]
        distances = list(filter(None, distances)) #Filter the pairs with not enough information
        distances = pd.DataFrame(distances, columns = ["userId", "userId2", "distance"])
        distances = distances.sort_values("distance")
        distances = distances.set_index("userId2")
        return distances.head(self.k)
    
    def print_recommendations(self,n=10):
        print(f"Here are the top {n} movies people like you enjoyed:")
        return self.__recommendations.drop(['neighbors_ratings','neighbors_count','mean_rating','total_votes','genres'],axis=1).head(n)
    

    def print_border(self,n=10):
        print(f"Here are the top {n} movies on the border of your bubble:")
        return self.__border_recommendations.drop(['neighbors_ratings','neighbors_count','mean_rating','total_votes','genres'],axis=1).head(n)

    def get_recommendations(self,n=10):
        return self.__recommendations.drop(['neighbors_ratings','neighbors_count','mean_rating','total_votes','genres'],axis=1).head(n)

    def get_border(self,n=10):
        return self.__border_recommendations.drop(['neighbors_ratings','neighbors_count','mean_rating','total_votes','genres'],axis=1).head(n)



class StochasticGradientDescentRecommender:

    def __init__(self, user,K=100,epochs=50):
        self.user = user.userId
        self.new_user_ratings = user.uratings_df
        self.new_user_ratings['userId'] = self.user
        self.new_user_ratings = self.new_user_ratings.reset_index()
        self.K = K
        self.n = 0
        self.sse = 0
        self.mse = 0
        self.epochs = epochs
        self.original_alpha = 0.01        #The learning rate
        self.alpha = self.original_alpha 
        self.gamma = 0.99                 #The learning rate schedule
        self.epochcount = 0
        self.epochcount = self.epochs
        self.empty_values_limit = 12
        self.top_users_limit = 25
        self.break_limit = 0.2
        
    # Method for start model processing.
    
    def run(self, user_ratings=None, print_epochs=True, evaluating=False,ignore_break=False):
        
        #Resets alpha in case of second+ run.
        self.alpha_setter(self.original_alpha)
        
        #This is to allow for both the first run and evaluation run.
        if type(user_ratings) == type(None): 
            user_ratings = self.new_user_ratings
            
        # Get pivot table.
        self.ratings_df, self.pivot_df = self.__dataframe_reduction(user_ratings, ratings, movies)
        
        
        # This loop limits the amount of empty values in the pivot table in order to facilitate a converging descent.
        # However, this also limits the number of movies available for recommendation.
        for column in self.pivot_df.columns:
            try:
                if self.pivot_df[column].value_counts()[0]>self.empty_values_limit:
                    self.pivot_df = self.pivot_df.drop(column, axis=1)
            except:
                pass
         
        # Run the matrix factorization and calculate the complete table
        self.new_P, self.new_Q = self.__SGD_MatFac(self.pivot_df,print_epochs,ignore_break)
        self.new_R = np.dot(self.new_P,self.new_Q.T)
        
        #Prepare recommendation tables
        self.__recommend(self.new_R, self.pivot_df)
        
        #if not evaluating:
        #    self.get_recommendations(n=10)

    #Calculates the pivot table based on user ratings. Due to the size of the original dataframes, the number of movies
    # and users considered is reduced to decrease processing time. 
    
    def __dataframe_reduction(self, user_ratings, ratings_df, movies_df): 
              
        ratings_df = ratings_df.drop(ratings_df[ratings_df["movieId"].isin(movies_df[movies_df.total_votes<20].index)].index)
        top_users = ratings.groupby(['userId'])['userId'].count()
        top_users = top_users.sort_values(ascending=False)
        top_users = top_users.head(self.top_users_limit)
        ratings_df = ratings_df[ratings_df['userId'].isin(top_users.index)]
        ratings_df = ratings_df.append(user_ratings)
        pivot_df = ratings_df.pivot(index='userId',columns='movieId', values='rating').fillna(0)
        return ratings_df, pivot_df

    
    # Stochastic Gradient Descent Matrix Factorization
    
    def __SGD_MatFac(self,R,print_epochs=True,ignore_break=False):
    
        print('Model run start.')
        if type(R) == pd.core.frame.DataFrame:
            R = R.values
            
        N = len(R)
        M = len(R[0])
        P = np.random.rand(N,self.K)*(5/self.K)
        Q = np.random.rand(M,self.K)*(5/self.K)
        Q=Q.T
        self.n = np.count_nonzero(R==0) #N for calculation of MSE/RMSE
        
        self.sample_index_list = []
        for i in range(N):
            for j in range(M):
                self.sample_index_list.append([i,j])
        np.random.shuffle(self.sample_index_list)
        
        
        for epoch in range(self.epochs):   
            e=0
            for i,j in self.sample_index_list:  
                if R[i][j]>0: #The error is only calculated for the known values.
                    eij = R[i][j]-np.dot(P[i,:],Q[:,j])
                    for k in range(self.K):
                        P[i][k] = P[i][k] + self.alpha * (eij * Q[k][j])
                        Q[k][j] = Q[k][j] + self.alpha * (eij * P[i][k])            
                    e = e+pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
            self.sse = e
            self.mse = self.sse/self.n
            self.rmse = np.sqrt(self.mse)
            if not ignore_break:
                if self.rmse<self.break_limit:
                    print('Breaking at Epoch: {}, SSE {:.2f}, RMSE: {:.4f}'.format(epoch,self.sse,self.rmse))
                    break
            if print_epochs:
                print('Epoch: {}, RMSE {:.4f}'.format(epoch,self.rmse))
            self.alpha_setter(self.alpha*self.gamma)
        print('Run complete.')
        return P, Q.T

    # Prepares recommendation matrix
    
    def __recommend(self, predicted_matrix, original_matrix):
        user_rated = original_matrix.loc[self.user]
        user_rated = user_rated.reset_index()
        user_predictions = pd.DataFrame(pd.DataFrame(predicted_matrix, index= original_matrix.index).loc[self.user])
        recommend_matrix = pd.DataFrame(user_rated).join(user_predictions,lsuffix='l', rsuffix='r').set_index('movieId')
        recommend_matrix.columns = ['original','predictions']
        self.recommend_matrix = recommend_matrix
        recommendations = recommend_matrix[recommend_matrix['original']==0].drop('original',axis=1).sort_values('predictions',ascending=False)
        recommendations_final = recommendations.join(movies).drop(['predictions','total_votes'],axis=1).head(10)
        self.recommendations_final = recommendations_final
        self.comparison_matrix = self.recommend_matrix[self.recommend_matrix.original!=0]
        
    # Setter for alpha (aka learning rate)
    
    def alpha_setter(self,new_alpha):
        self.alpha=new_alpha
        
    # Method to continue processing the model with additional epochs. 
    
    def continue_optimization(self,more_epochs,print_epochs=True,ignore_break=False):
        
        print('Restarting model.')
        R = self.pivot_df.values
        N = len(R)
        M = len(R[0])
        P = self.new_P
        Q = self.new_Q
        Q=Q.T
        
        
        for epoch in range(more_epochs): 
            e=0
            for i,j in self.sample_index_list:  
                    if R[i][j]>0: #The error is only calculated for the known values.
                        eij = R[i][j]-np.dot(P[i,:],Q[:,j])
                        for k in range(self.K):
                            P[i][k] = P[i][k] + self.alpha * (eij * Q[k][j])
                            Q[k][j] = Q[k][j] + self.alpha * (eij * P[i][k])
                        e = e+pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
            self.sse = e
            self.mse = self.sse/self.n
            self.rmse = np.sqrt(self.mse)
            if not ignore_break:
                if self.rmse<self.break_limit:
                    print('Breaking at epoch: {}, SSE {:.2f}, RMSE: {:.4f}'.format(epoch,self.sse,self.rmse))
                    break
            if print_epochs:
                print('Epoch: {}, RMSE {:.4f}'.format(self.epochcount,self.rmse))
            self.epochcount+=1
            self.alpha_setter(self.alpha*self.gamma)
        
        self.new_P = P
        self.new_Q = Q.T
        self.new_R = np.dot(self.new_P,self.new_Q.T)
        self.__recommend(self.new_R, self.pivot_df)
        print("Run complete.")
   
    # Prints results.
    
    def get_recommendations(self,n=10):        
        print("-------------------------------------------")
        print("Based on our community votes database, you may like:")
        return self.recommendations_final.head(n)
    
    # Creates evaluation matrix by separting the available user ratings in "train" and "test" splits.
    
    def create_evaluation_matrix(self, sample_size,print_epochs,ignore_break=False):
        ratings_sample = self.new_user_ratings.sample(int(self.new_user_ratings.shape[0]*sample_size),random_state=42)['movieId']
        test_ratings = self.new_user_ratings.copy()
        for i in ratings_sample.values:
            test_ratings.loc[test_ratings['movieId'] == i,'rating']=0
        self.run(test_ratings, print_epochs=print_epochs,evaluating=True,ignore_break=ignore_break)
        self.evaluation_matrix = self.recommend_matrix[self.recommend_matrix['original']==0].sort_values('predictions',ascending=False)
        self.evaluation_matrix  = self.evaluation_matrix.join(self.new_user_ratings.set_index('movieId'), how='left').drop(['userId','original'],axis=1).dropna()       
        ##### Interpolation may be used if the values are not expanding towards the extremes (0 and 5).
#         interpolation_list = self.evaluation_matrix['predictions'].to_numpy()
#         self.evaluation_matrix['predictions'] = np.interp(interpolation_list, (interpolation_list.min(),interpolation_list.max()),(0,5))
        #####
        self.evaluation_values = self.evaluation_matrix.values
      
    # Method to evaluate the model based on Precision@k and Recall@k.
    
    def evaluate_model(self, relevancy=3.5, sample_size=0.5, epochs=50,print_epochs=False,ignore_break=False):                
        self.alpha_setter(self.original_alpha)
        self.epochs = epochs
        self.create_evaluation_matrix(sample_size,print_epochs,ignore_break)
        TP, FP, FN, TN = [0,0,0,0]
        k = int(self.new_user_ratings.shape[0]*sample_size)
        for i in range(len(self.evaluation_values)):
            if self.evaluation_values[i, 0]>=relevancy and self.evaluation_values[i, 1]>=relevancy:
                TP+=1
            elif self.evaluation_values[i, 0]>=relevancy and self.evaluation_values[i, 1]<relevancy:
                FN+=1
            elif self.evaluation_values[i, 0]<relevancy and self.evaluation_values[i, 1]>=relevancy:
                FP+=1
            else:
                TN+=1
            try:
                if i==9:
                    recall_at_10 = TP/(TP+FN)
                    precision_at_10 = TP/(TP+FP)
            except:
                pass
                
        recall_at_k = TP/(TP+FN)
        precision_at_k = TP/(TP+FP)
        print("-------------------------------------------")
        print("K: {},TP: {}, FP: {}, FN: {}, TN:{}".format(k, TP,FP,FN,TN))
        print("For a relevancy of {}, our model has a Precision@{} of {:.2f}% and a Recall@{} of {:.2f}%".format(relevancy, k, precision_at_k*100, k, recall_at_k*100))
        if k>15:
            try:
                print("For a relevancy of {}, our model has a Precision@10 of {:.2f}% and a Recall@10 of {:.2f}%".format(relevancy, precision_at_10*100, recall_at_10*100))
            except:
                pass
    
    # Returns the highest prediction value. Also, if there are too many empty
    # values in the original pivot table, the probabillity of one of the values of the complete table being too high
    # increases, due to a random chance of the dot product of the matrices containg the latent features not being
    # properly optmized.
    
    def get_max_prediction(self):
        print(f"Max user prediction {self.recommend_matrix[self.recommend_matrix['original']==0]['predictions'].max()}")
        print(f"Max general prediction {self.new_R.max()}")