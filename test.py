import numpy as np
ratingsPath = ".\\data\\ratings.csv"
ratingsDF = pd.read_csv(ratingsPath, index_col=None)

trainRatingsPivotDF = pd.pivot_table(ratingsDF[['userId', 'movieId', 'rating']], columns=['movieId'],
                                 index=['userId'], values='rating', fill_value=0)
ratingValues = trainRatingsPivotDF.values.tolist()
usersMap = dict(enumerate(list(trainRatingsPivotDF.index)))
print(usersMap[1])