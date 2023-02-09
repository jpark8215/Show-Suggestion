import numpy as np
import pandas as pd
# from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# import data
data = pd.read_csv('kdrama.csv')
data_arr = np.array(data)
# print(X)

data = data[["Name",
             # "Aired Date",
             "Year of release",
             "Original Network",
             "Aired On",
             # "Number of Episodes",
             # "Duration",
             "Content Rating",
             "Rating",
             "Synopsis",
             "Genre",
             "Tags",
             "Director",
             "Screenwriter",
             "Cast",
             "Production companies",
             # "Rank"
             ]]
data.head()
# print(data)

# to vector
# text_data = data_arr
# vector = SentenceTransformer('distilbert-base-nli-mean-tokens')
# embeddings = vector.encode(text_data, show_progress_bar=True)

# create the transform
vectorizer = CountVectorizer(stop_words='english')
# vectorizer = CountVectorizer()
# tokenize and build vocab
vectorizer.fit(data)
# summarize
# print("Vocabulary: ", vectorizer.vocabulary_)
# encode data
vector = vectorizer.transform(data)
# summarize encoded vector
# print("Encoded data: \n", vector.toarray())
# df = pd.DataFrame(vector.toarray())
# sns.pairplot(df)
# plt.show()

# compute the cosine similarity and the function
cos_sim_data = pd.DataFrame(cosine_similarity(vector))


print("cos sim data: \n", cos_sim_data)


def give_recommendations(index, print_recommendation=False, print_recommendation_synopsis=False, print_genres=False):
    index_recomm = cos_sim_data.loc[index].sort_values(ascending=False).index.tolist()[1:3]
    show_recomm = data['Name'].loc[index_recomm].values
    result = {'Shows': show_recomm, 'Index': index_recomm}
    if print_recommendation:
        print('You watched: %s \n' % (data['Name'].loc[index]))
        k = 1
        for show in show_recomm:
            print('Recommendation %i : %s \n' % (k, show))
            k = k + 1
    if print_recommendation_synopsis:
        print('The synopsis of the show you watched is :\n %s \n' % (data['Synopsis'].loc[index]))
        k = 1
        for q in range(len(show_recomm)):
            plot_q = data['Synopsis'].loc[index_recomm[q]]
            print('The synopsis of the recommendation %i is :\n %s \n' % (k, plot_q))
            k = k + 1
    if print_genres:
        print('The genres of the show you watched is(are) :\n %s \n' % (data['Genre'].loc[index]))
        k = 1
        for q in range(len(show_recomm)):
            plot_q = data['Genre'].loc[index_recomm[q]]
            print('The genres of the recommendation %i is(are) :\n %s \n' % (k, plot_q))
            k = k + 1
    return result


# plot recommendation
for q in range(1, 3):
    plt.subplot(2, 1, q)
    index = np.random.choice(np.arange(0, len(cos_sim_data)))
    to_plot_data = cos_sim_data.drop(index, axis=1)
    plt.plot(to_plot_data.loc[index], '.', color='blue')
    recomm_index = give_recommendations(index)
    x = recomm_index['Index']
    y = cos_sim_data.loc[index][x].tolist()
    m = recomm_index['Shows']
    plt.plot(x, y, '.', color='black', label='Recommended Shows')
    plt.title('Show Watched: \n' + data['Name'].loc[index])
    plt.xlabel('Show Index')
    k = 0
    for x_i in x:
        plt.annotate('%s' % (m[k]), (x_i, y[k]), fontsize=10)
        k = k + 1

    plt.ylabel('Cosine Similarity')
    plt.show()

# given watched show
give_recommendations(1, True, True, True)
