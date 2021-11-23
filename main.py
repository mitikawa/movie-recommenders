from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
from utils import BestRatedRecommender, StochasticGradientDescentRecommender, bubble_rec, RarePearlsRecommender

movies = pd.read_csv("data/movies.csv")
movies = movies.set_index("movieId")
movies_images = pd.read_csv("data/movies_with_images.csv", index_col='movieId')
ratings = pd.read_csv("data/ratings.csv")
ratings = ratings.drop("timestamp", axis=1)
movies['mean_rating'] = ratings.groupby("movieId").mean()['rating']
movies['total_votes'] = ratings["movieId"].value_counts()
movies = movies.dropna()  # Drop movies that don't appear in the ratings dataframe
last_user = ratings['userId'].max()

#most_popular = movies_images.sort_values(by=["total_votes"],ascending=False).copy()
#most_popular = most_popular.head(20)

#movieIds = list(most_popular.index)
#movieNames = list(most_popular.title)
#movieImages = list(most_popular.link)

movieIds = [356, 318, 6016, 2571, 260, 858, 7153, 35836, 4306, 1214, 595, 79132, 99114, 1704, 7361, 2706,
            5816, 527, 2959, 364, 6874, 586, 1923, 1258, 44191, 1387, 6373, 1270, 74458, 68157, 5618, 49272, 51662]
movieNames = list(movies_images.loc[movieIds].title)
movieImages = list(movies_images.loc[movieIds].link)

first_visit = True


class User:
    def __init__(self):
        self.userId = last_user+1  # UserId
        self.ratings_to_show = []
        self.uratings = []
        self.umovieids = []

    def add_movie(self, id, rating):
        self.ratings_to_show.append(rating)
        self.uratings.append(float(rating)/2)
        self.umovieids.append(int(id))

    def update_movie(self, id, rating):
        loc = self.umovieids.index(id)
        self.ratings_to_show[loc] = rating
        self.uratings[loc] = float(rating)/2

    def update_df(self):
        self.uratings_df = pd.DataFrame({'movieId': self.umovieids,
                                         'rating': self.uratings}).set_index('movieId')


app = Flask(__name__)
app.secret_key = 'alura'
my_user = User()


@app.route('/')
def landing_page():
    if not first_visit:
        return redirect(url_for('home'))
    return render_template('landingPage.html', titulo='Welcome!')


@app.route('/about')
def about():
    if not first_visit:
        return redirect(url_for('home'))
    return render_template('about.html', titulo='About this project.')


@app.route('/full_about')
def full_about():
    return render_template('full_about.html', titulo='About this project.')


@app.route('/start')
def start():
    if not first_visit:
        return redirect(url_for('home'))
    return render_template('start.html', titulo='Adding user ratings', movies=movieNames, links=movieImages)


@app.route('/create', methods=['POST', ])
def create():
    selections = {}
    for i in range(1, len(movieNames)+1):  # to change
        selections[i] = request.form['movie-'+str(i)]
        my_user.add_movie(movieIds[i-1], selections[i])
    my_user.update_df()
    global first_visit
    first_visit = False
    return redirect(url_for('home'))


@app.route('/home')
def home():
    if first_visit:
        return redirect(url_for('landing_page'))
    return render_template('home.html', titulo='Home', ratings=my_user.ratings_to_show, movies=list(movies.loc[my_user.umovieids].title))

@app.route('/add')
def add():
    if first_visit:
        return redirect(url_for('home'))
    return render_template('add_rating.html', titulo='Adding user ratings', allMovieIds=list(movies_images.index), allMovieNames=list(movies_images.title))

@app.route('/add_to_user', methods=['POST', ])
def add_to_user():
    if first_visit:
        return redirect(url_for('home'))
    movie_selection = int(request.form['movies'])
    rating_selection = int(request.form['ratings'])
    if movie_selection not in my_user.umovieids:
        my_user.add_movie(movie_selection, rating_selection)
        my_user.update_df()
        flash('Movie added.')
        return redirect(url_for('home'))
    else:
        flash('Movie was already rated with rating {}, rating updated to {}.'.format(my_user.ratings_to_show[my_user.umovieids.index(movie_selection)],rating_selection))
        my_user.update_movie(movie_selection, rating_selection)
        my_user.update_df()
        return redirect(url_for('home'))

@app.route('/top')
def top_rated():
    if first_visit:
        return redirect(url_for('landing_page'))
    BRR = BestRatedRecommender(my_user.uratings_df)
    BRR_recommendations = BRR.recommend(50)
    return render_template('recommendation_page.html', titulo='Top Rated Movies', ids=list(BRR_recommendations.index), movies=list(BRR_recommendations.title))


@app.route('/indies')
def indies():
    if first_visit:
        return redirect(url_for('landing_page'))
    RPR = RarePearlsRecommender(my_user.uratings_df)
    RPR_recommendations = RPR.recommend(50)
    return render_template('recommendation_page.html', titulo='Indies', ids=list(RPR_recommendations.index), movies=list(RPR_recommendations.title))


@app.route('/bubble')
def bubble():
    if first_visit:
        return redirect(url_for('landing_page'))
    rec = bubble_rec(user=my_user, k=100)
    recommendations = rec.get_recommendations()

    return render_template('recommendation_page.html', titulo='Bubble Recommender', ids=list(recommendations.index), movies=list(recommendations.title))


@app.route('/border')
def border():
    if first_visit:
        return redirect(url_for('landing_page'))
    rec = bubble_rec(user=my_user, k=100)
    border = rec.get_border()

    return render_template('recommendation_page.html', titulo='Border Recommender', ids=list(border.index), movies=list(border.title))


@app.route('/smart')
def smart():
    if first_visit:
        return redirect(url_for('landing_page'))
    sgd = StochasticGradientDescentRecommender(my_user, K=100, epochs=50)
    sgd.run()
    sgd_rec = sgd.get_recommendations(n=10)

    return render_template('recommendation_page.html', titulo='Smart Recommender', ids=list(sgd_rec.index), movies=list(sgd_rec.title))


app.run(debug=True)
