import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, TruncatedSVD
from afinn import Afinn
from imblearn.under_sampling import RandomUnderSampler
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import en_core_web_sm

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.metrics import confusion_matrix
from sklearn.base import clone
from scipy.spatial.distance import euclidean, cityblock
from itertools import cycle

from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud

import spacy
import nltk
import unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem.snowball import SnowballStemmer
import re
from nltk import FreqDist

tokenizer = ToktokTokenizer()
nlp = en_core_web_sm.load()
nltk.download('stopwords', quiet=True)
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.append('nt')

def get_data():
    """Return the dataframe that contains the reviews."""

    df = pd.read_csv('/mnt/data/public/amazon-reviews/'
            'amazon_reviews_us_Digital_Ebook_Purchase_v1_01.tsv.gz',
             compression= 'gzip', sep='\t', error_bad_lines=False)
  
    df = df.dropna()
    df['review_date']=pd.to_datetime(df['review_date'])
    df['star_rating'] = df['star_rating'].astype(int)
    
    return df


def stardist_plotter(df, f_num):
    """Plot the star ratings distribution."""
    
    fig = plt.figure(figsize = (10,5))
    ax = fig.gca()
    ax.set_title('Figure '+str(f_num)+'. Distribution of the'
                 ' star rating feature.', fontsize=16)
    ax.set_xlabel('Number of ratings', fontsize=14)
    ax.set_ylabel('Ratings', fontsize=14)
    df['star_rating'].hist(orientation='horizontal',
                           ax=ax, color='b', grid=False);

    
def helpful_votes_distplotter(df, f_num):
    """Plot the distribution of helpful votes."""
    
    fig = plt.figure(figsize = (10,5))
    ax = fig.gca()
    ax.grid(False)
    ax.set_title('Figure '+str(f_num)+'. Distribution of the'
                 ' helpful votes feature.', fontsize=16)
    ax.set_xlabel('Count of helpful votes', fontsize=14)
    ax.set_ylabel('Number of reviews', fontsize=14)
    df['helpful_votes'].hist(ax=ax, range=[0, 20],
                             color='b', grid=False);
    
    
def total_votes_distplotter(df, f_num):
    """Plot the distribution of total votes."""
    
    fig = plt.figure(figsize = (10,5))
    ax = fig.gca()
    ax.grid(False)
    ax.set_title('Figure '+str(f_num)+'. Distribution of the'
                 ' helpful votes feature.', fontsize=16)
    ax.set_xlabel('Count of total votes', fontsize=14)
    ax.set_ylabel('Number of reviews', fontsize=14)
    df['total_votes'].hist(ax=ax, range=[0, 20],
                          color='b', grid=False);

    
def pearson_heatmap(df, f_num):
    """Plot pearson correlational heatmap."""
    
    fig = plt.figure(figsize = (10,5))
    ax = fig.gca()
    ax.set_title('Figure '+str(f_num)+'. Feature'
                 ' Correlational Heatmap', fontsize=16)
    sns.heatmap(df.corr());

                
def plot_year_reviews(df, f_num):
    """Plot the distributions of reviews per year."""
    
    plt.figure(figsize = (10,5))
    sns.displot(df['review_date'], height=5, aspect=1.75, color='b')
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Counts', fontsize=14)
    plt.title('Figure '+str(f_num)+'. Number of Reviews per year',
              fontsize=16);

    
def year_customer_reviews(df, f_num):  
    """Plot customer counts and reviews per year."""
    
    df['year']=df['review_date'].dt.year    
    dummy = df['review_date'].dt.year.value_counts()
    df2 = pd.DataFrame(dummy).reset_index()
    df2.rename(columns={'index':'year', 'review_date':'Review counts'}, inplace=True)
    dummy = pd.DataFrame(df.groupby('year')['customer_id'].nunique()).reset_index()
    df3 = df2.merge(dummy, on='year')
    plt.figure(figsize=(10,5))

    plt.plot(df3['year'], df3['Review counts'], 'o-', color='b',
             label='Review Counts')
    plt.plot(df3['year'], df3['customer_id'], 'o-', color='orange',
            label='Customer Counts')

    plt.xlabel('year', fontsize=14)
    plt.ylabel('counts', fontsize=14)
    plt.title('Figure '+str(f_num)+'. Total number of reviews'
              ' and customers from year 1999-2013',
              fontsize=16)
    plt.legend(bbox_to_anchor=(1,1))

    
def bar_plot_year(df, f_num):
    """Plot the distribution of reviews per year."""
    
    fig = plt.figure(figsize = (10,5))
    ax = fig.gca()
    ax.set_title('Figure '+str(f_num)+'. Number of'
                 ' Reviews per year', fontsize=16)
    ax.set_xlabel('Number of reviews', fontsize=14)
    ax.set_ylabel('Year', fontsize=14)
    df['year'].value_counts().plot(kind='barh',
                                   color='b',ax=ax);

    
def grouper(df, year, f_num, twozeroten_mod=False):
    """Returns grouped data points per year and top 10 books per year."""
    
    grouped_years = df.groupby('year')
    if twozeroten_mod == False:
        fig = plt.figure()
        ax = fig.gca()
        ax.set_title('Figure '+str(f_num)+'. '+
                     'Top 10 books based on ratings and purchase/review'
                     ' count for '+str(year), 
                     fontsize=14)
        ax.set_xlabel('Number of reviews/purchases', fontsize=12)
        df_yr = grouped_years.get_group(year)
        df_yr_5 = df_yr[df_yr['star_rating'] == 5]
        df_yr_5['product_title'].value_counts().head(10).plot(kind='barh', 
                                                    color='b',ax=ax);
    else:
        fig = plt.figure()
        ax = fig.gca()
        ax.set_title('Figure '+str(f_num)+'. Top 10 books based on ratings'
                     ' and purchase/review count for 2010', 
                     fontsize=14)
        ax.set_xlabel('Number of reviews/purchases', fontsize=12)
        df_yr = grouped_years.get_group(2010)
        df_yr_5 = df_yr[df_yr['star_rating'] == 5]
        df_2010_5_sol = df_yr_5[(df_yr_5['star_rating'] == 5) & \
                        (df_yr_5['product_title'] != \
                         '2010 ABNA Quarterfinalist 11')]
        df_2010_5_sol['product_title'].value_counts().head(10)\
        .plot(kind='barh', ax=ax, color='b');
        
        
def truncate_df(start_date, df):
    """Return a subset of the amazon e-book data
    given `start_date`."""
    
    df = df[df['review_date'] >= start_date]
    return df


def remove_accented_chars(text):
    """Convert and standardize text into ASCII characters. It made
    sure characters which look identical actually are identical.
    Lastly, convert text into small letters.
    """
    
    text = unicodedata.normalize('NFKD', text).encode('ascii',
                              'ignore').decode('utf-8', 'ignore')
    return text.lower()


def remove_stopwords(text, is_lower_case=False):
    """Return the filtered text where stopwords are removed."""

    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in
                           stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() 
                           not in stopword_list]
    text = ' '.join(filtered_tokens)
    return text


def remove_special_characters(text, remove_digits=True):
    """Return the filtered text where special caharacters
    are removed."""
        
    text= re.sub(r'[\r|\n|\[|\]]+', '',text)
    text = text.replace('\\\\', '')
    #remove symbols
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    text = re.sub('br', '', text)
    text = re.sub(' +', ' ', text)
    return text
    
    
def lemmatize_text(text):
    """Return the the root forms of the text."""

    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-'
                     else word.text for word in text])
    return text


def process_doc(doc):
    """Return the processed text that will be used in the
    analysis."""
    
    doc = str(doc)
    doc = remove_accented_chars(doc)
    doc = remove_stopwords(doc)
    doc = remove_special_characters(doc)
    doc = lemmatize_text(doc)
    return doc


def get_sentiment_df(df, start_date='2013-01-01',
                     n=20000, target_col='review_headline'):
    """Return a dataframe with clean text and `score_rating`."""
    
    df = truncate_df(start_date, df)
    X, y = random_equal_sampler(df, n)
    df1 = X.copy()
    df1['cleaned_'+target_col] = df1[target_col].apply(lambda x:
                                                    process_doc(x))
    df1['star_rating'] = y
    df1['score_group'] = -1
    df1.loc[df1['star_rating'].isin([4,5]), 'score_group'] = 1 
    df1.loc[df1['star_rating'].isin([3]), 'score_group'] = 0 
    return df1


def random_equal_sampler(df, n, random_state=0):
    """Return the random subset of Dataframe with n samples for each
    star rating.
    Parameters
    ----------
    df: pandas DataFrame
        The dataframe that contains the reviews.
        
    n : int
        The sample size for each class.
        
    random_state : int
        Control the randomization of the algorithm
    
    Returns
    --------
    X_resampled : {array-like, dataframe, sparse matrix} 
                The array containing the resampled data.

    y_resampled: array-like
                The corresponding label of X_resampled.
    """

    strategy = {1:n, 2:n, 3:n, 4:n, 5:n}
    rus = RandomUnderSampler(random_state=random_state,
                             sampling_strategy= strategy)
    X = df[['product_title','review_headline','review_body',
           'helpful_votes']]
    y = df['star_rating'].values
    rus.fit(X, y)
    X_resampled, y_resampled = rus.fit_resample(X, y )
    return X_resampled, y_resampled


def plot_sentiments_count(df, f_num):
    """Plot the distribution of sentiments of the dataset."""
    
    f_num = str(f_num)
    plt.figure(figsize = (14,5))
    negative_reviews = df[df['score_group'] ==-1]
    positive_reviews = df[df['score_group'] == 1]
    neutral_reviews = df[df['score_group'] == 0]
    
    sns.barplot(x =['positive', 'neutral', 'negative'],
                y=[len(positive_reviews),len(neutral_reviews),
                   len(negative_reviews)],data = df)
    
    plt.xlabel("Sentiments", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.title("Figure "+f_num+'. Sentiment Counts',fontsize=16)
    plt.show()
    
    
def plot_freq_words(df, sentiment, f_num):
    """Plot the Top 50 terms of the sentiment groups."""
    
    f_num = str(f_num)
    if str.lower(sentiment) == 'positive':
        reviews = df[df['score_group'] == 1]
        
    elif str.lower(sentiment) == 'negative':
        reviews = df[df['score_group'] ==-1]
        
    elif str.lower(sentiment) == 'neutral':
        reviews = df[df['score_group'] == 0]
    

    
    freq_dist = FreqDist([word for review in 
                      reviews['cleaned_review_headline']
                       for word in str(review).split()])
    plt.figure(figsize = (14,6))
    plt.title('Figure '+f_num+ '. Cumulative counts of '+
              'Top 50 most frequent words belonging to '+
              sentiment+' Sentiment', fontsize=16)
    
    plt.xlabel('Word', fontsize=14)
    plt.ylabel('Cumulative Counts', fontsize=14)
    freq_dist.plot(50, cumulative = True);
   
    
def word_cloud(df, sentiment, f_num, afinn=False):
    """Plot the wordcloud of the sentiments using the star
    rating groups from the reviews."""
    
    f_num = str(f_num)
    if afinn:
        reviews = df[df['sentiment_category'] == \
                     str.lower(sentiment)]
    else:  
        if str.lower(sentiment) == 'positive':
            reviews = df[df['score_group'] == 1]
        elif str.lower(sentiment) == 'negative':
            reviews = df[df['score_group'] ==-1]
           
        elif str.lower(sentiment) == 'neutral':
            reviews = df[df['score_group'] == 0]

    color = {'Negative': {'colormap' :'viridis',
                           'background_color' : 'black'},
              'Positive': {'colormap' :'viridis',
                           'background_color' : 'white'},
              'Neutral': {'colormap' :'plasma',
                           'background_color' : 'white'}}
    
    
    text  =  ' '.join([word for review in
                       reviews['cleaned_review_headline']
                       for word in review.split()])
    
    wordcloud = WordCloud(background_color=color[sentiment]\
                         ['background_color'], 
                        stopwords = stopword_list, 
                        colormap = color[sentiment]['colormap'],
                        max_words = 100, 
                       collocations=False).generate(text)
    plt.figure(figsize = (8,6), dpi = 200)
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis("off")
    if afinn:
        plt.title("Figure "+f_num+'. AFINN Word Count belonging to '
                  + sentiment+' Sentiment', fontsize=12)
    else:
        plt.title('Figure '+f_num+'. Word Cloud '
              'belonging to '+ sentiment+' Sentiment', fontsize=12)
    plt.show()
    
    
def get_afinn_score(df, plot=True, f_num=0):
    """Return a dataframe with new sentiment score using AFINN
    Sentiment scores.
    
    If plot is True, then plot the distribution of the
    Afinn sentiment groups."""
    f_num = str(f_num)
    corpus = df['cleaned_review_headline'].values
    af = Afinn() #initializing Afinn
    
    #Generating scores for every review ( from 0 to +5)
    sentiment_scores = [af.score(review) for review in corpus]
    #generating categories
    sentiment_category = ['positive' if score > 0\
                         else 'negative' if score < 0\
                         else 'neutral' \
                         for score in sentiment_scores]

    #Plotting the sentiment group counts
    df['sentiment_score'] = sentiment_scores
    df['sentiment_category'] = sentiment_category
    negative_reviews = df[df['sentiment_category'] == 'negative']
    positive_reviews = df[df['sentiment_category'] == 'positive']
    neutral_reviews = df[df['sentiment_category'] == 'neutral']
    
    if plot:
        plt.figure(figsize = (14,5))
        sns.barplot(x = df['sentiment_category'].unique(),
                    y=[len(neutral_reviews),
                        len(positive_reviews),
                       len(negative_reviews)], data = df)
        plt.xlabel("Sentiments", fontsize=14)
        plt.ylabel("Counts", fontsize=14)
        plt.title("Figure "+f_num+'. AFINN Sentiment Counts',
                  fontsize=16)
        plt.show()
    return df


def project_svd(q, s, k):
    """Accept q, s and k and return the design matrix projected on to the
    first k singular vectors.
    """
    return q[:,:k].dot(s[:k,:k])


def plot_svd(X_new, features, p, f_num):
    """
    Plot transformed data and features on to the first two singular vectors
    
    Parameters
    ----------
    X_new : array
        Transformed data
    featurs : sequence of str
        Feature names
    p : array
        P matrix
    """
    fig, ax = plt.subplots(1, 2, subplot_kw=dict(aspect='equal'), 
                           gridspec_kw=dict(wspace=0.4), dpi=150)
    ax[0].scatter(X_new[:,0], X_new[:,1])
    ax[0].set_xlabel('SV1', fontsize=8)
    ax[0].set_ylabel('SV2', fontsize=8)

    for feature, vec in zip(features, p):
        ax[1].arrow(0, 0, vec[0], vec[1], width=0.01, ec='none', fc='r')
        ax[1].text(vec[0], vec[1], feature, ha='center', color='r', fontsize=5)
    ax[1].set_xlim(-1, 1)
    ax[1].set_ylim(-1, 1)
    ax[1].set_xlabel('SV1', fontsize=8)
    ax[1].set_ylabel('SV2', fontsize=8)
    plt.suptitle('Figure '+str(f_num)+'. Projection of features on'
                 ' SV1 and SV2',fontsize=10)

    
def plot_variance_per_n(tfidf, n_components=429,threshold_variance=0.80,
                        plot=True, f_num=None):
    """Return the optimal n_components `k` and the model `lsa`
    after performing the truncated SVD to the tfidf matrix.
    
    If plot is True, then plot the variance explained
    per n_components of the singular values SVs.
    
    Parameters
    ---------
    tfidf : array-like or matrix
        The data to be used in the analysis.
    n_components : int
        The number of components that will be used in the svd.
    threshold_variance : int
        The threshold variance that will be set in the analysis.
    plot : boolean
        If plot is True, then plot the variance explained
        per n_components of the singular values SVs.
    f_num : int
        Figure number to be annotated in the figure title.
    """
    
    lsa = TruncatedSVD(n_components=n_components)
    doc_topic = lsa.fit_transform(tfidf)
    variance_explained_ratio = lsa.explained_variance_ratio_
    df = pd.DataFrame(variance_explained_ratio.cumsum())
    k = df[df[0] > threshold_variance].index[0]
    
    if plot:
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(range(1, len(variance_explained_ratio)+1), 
                variance_explained_ratio,'-', label='individual')
        ax.set_xlim(0, len(variance_explained_ratio)+1)
        ax.set_xlabel('SV', fontsize=14)
        ax.set_ylabel('variance explained', fontsize=14)
        ax = ax.twinx()
        ax.plot(range(1, len(variance_explained_ratio)+1), 
                variance_explained_ratio.cumsum(), 'r-', label='cumulative')
        ax.axhline(threshold_variance, ls='--', color='g')
        ax.axvline(k, ls='--', color='g')
        ax.set_ylabel('cumulative variance explained')
        ax.set_title('Figure '+str(f_num)+'. Variance explained for n SVs',
                     fontsize=16);
    return k, lsa, doc_topic


def plot_SV_vs_features(doc_topic, tfidf, model, feature_names, f_num=20):
    """Return topic vector `VT` and plot the projected features
    onto SV1 and SV2."""
    
    q = doc_topic / model.singular_values_
    sigma = np.diag(model.singular_values_)
    VT = model.components_
    U = model.transform(tfidf) / model.singular_values_
    X_new = project_svd(q, sigma, k=131)
    plot_svd(X_new, feature_names, VT, f_num)
    return VT


def get_matrix_tfidf(df):
    """Apply TFIDF to the data into and return the tfidf matrix and
    dataframe format."""
    
    corpus = df['cleaned_review_headline'].values
    TFIDF = TfidfVectorizer(min_df=20,lowercase=True)
    tfidf = TFIDF.fit_transform(corpus)
    feature_names = TFIDF.get_feature_names()
    df1 = pd.DataFrame(tfidf.toarray(), columns=TFIDF.get_feature_names())
    return tfidf, df1, feature_names


def plot_topics(Vt, feature_names, f_num=2):
    """Plot the top 10 topics using LSA model.Use only
    the first 6 SVs."""
    
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(14,8))
    plt.suptitle('Figure '+str(f_num)+'. Top 10 Topics correlated with the'
                 ' first 6 SVs', fontsize=18)

    for i, ax in enumerate(axs.reshape(-1)): 
        order = np.argsort(np.abs(Vt[:, i]))[-10:]
        ax.barh([feature_names[o] for o in order], Vt[order, i])
        ax.set_title(f'SV{i+1}', fontsize=15)
        if i in [0,3]:
            ax.set_ylabel(str('Topic'), fontsize=15)
        ax.set_xlabel(str('Weight'), fontsize=15)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


def display_topics(model, feature_names, no_top_words, topic_names=None,
                   n_topic=10):
    """Display the top 10 topics using the `model`."""
    
    for ix, topic in enumerate(model.components_[:n_topic]):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)   
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

        
def reduce_tfidf_matrix(k, X, feature_names, df1):
    """Return a dataframe `X_new` of the reduce tfidf matrix with k components,
    `y` actual sentiment labels.

    The reduced tfidf should be merged with `helpful_votes` found in df1."""

    lsa = TruncatedSVD(n_components= k)
    X = lsa.fit_transform(X)
    
    features = [feature_names[i] for i in lsa.components_[0].argsort()[::-1]]
    feature_names = features[:k]

    lst1 = X.tolist()
    df = pd.DataFrame(lst1, columns=feature_names)
    
    X_new = pd.merge(df, df1['helpful_votes'], right_index=True,
                     left_index=True)
    y = df1['score_group']

    return X_new, y
        

def kmeans_cluster_search(X, f_num=23):
    """Implement KMeans on varying number of k clusters"""
    X = pd.DataFrame(X)
    fig, ax = plt.subplots(2, 5, dpi=150, sharex=True, sharey=True, 
                           figsize=(8,5),
                           subplot_kw=dict(aspect='equal'),
                           gridspec_kw=dict(wspace=0.01));
    for i in range(2, 12):
        kmeans = KMeans(n_clusters=i, random_state=1337)
        y = kmeans.fit_predict(X)
        if i < 7:
            ax[0][i%7-2].scatter(X[0], X[1], s=12, c=y, alpha=0.5, 
                                 marker='.');
            ax[0][i%7-2].set_title('$k=%d$'%i);
        else:
            ax[1][i%7].scatter(X[0], X[1], s=12, c=y, alpha=0.5, marker='.');
            ax[1][i%7].set_title('$k=%d$'%i)
            
    plt.suptitle('Figure '+str(f_num)+'. K-Means clustering for different'
                 ' k using the First and Second Singular Values',
                 fontsize=16)
    plt.show()


def pooled_within_ssd(X, y, centroids, dist):
    """Compute pooled within-cluster sum of squares around the cluster mean
    
    Parameters
    ----------
    X : array
        Design matrix with each row corresponding to a point
    y : array
        Class label of each point
    centroids : array
        Cluster centroids
    dist : callable
        Distance between two points. It should accept two arrays, each 
        corresponding to the coordinates of each point
        
    Returns
    -------
    float
        Pooled within-cluster sum of squares around the cluster mean
    """
    ni = np.bincount(y.astype(int))
    return sum(dist(x_i, centroids[y_i])**2 / (2*ni[y_i])
                  for x_i, y_i in zip(X, y.astype(int)))


def purity(y_true, y_pred):
    """Compute the class purity
    
    Parameters
    ----------
    y_true : array
        List of ground-truth labels
    y_pred : array
        Cluster labels
        
    Returns
    -------
    purity : float
        Class purity
    """
    # Cluster purity: Measures how dominant the dominant 
    # class of each ground truth class is
    cm = confusion_matrix(y_true, y_pred)
    P_j = cm.max(axis=0) # P_j in the dominant class of cluster j
    M_j = cm.sum(axis=0) # M_j the number of data points in cluster j, 
                         # kt is the number of ground truth clusters
    Purity = P_j.sum() / M_j.sum() # High values of Purity are desirable.
    return Purity


def gap_statistic(X, y, centroids, dist, b, clusterer, random_state=None):
    """Compute the gap statistic
    
    Parameters
    ----------
    X : array
        Design matrix with each row corresponding to a point
    y : array
        Class label of each point
    centroids : array
        Cluster centroids
    dist : callable
        Distance between two points. It should accept two arrays, each 
        corresponding to the coordinates of each point
    b : int
        Number of realizations for the reference distribution
    clusterer : KMeans
        Clusterer object that will be used for clustering the reference 
        realizations
    random_state : int, default=None
        Determines random number generation for realizations
        
    Returns
    -------
    gs : float
        Gap statistic
    gs_std : float
        Standard deviation of gap statistic
    """
    rng = np.random.default_rng(random_state)
    
    log_wki = []
    log_w = np.log(pooled_within_ssd(X, y, centroids, dist))
    
    for i in range(b):
        X_norm = rng.uniform(X.min(axis=0), X.max(axis=0), size=X.shape)
        y_predict = clusterer.fit_predict(X_norm)
        centroid_norm = clusterer.cluster_centers_
        w_ki = pooled_within_ssd(X_norm, y_predict, centroid_norm, dist)
        log_wki.append(np.log(w_ki))    
    
    return (log_wki - log_w).mean(), (log_wki - log_w).std()


def cluster_range(X, clusterer, k_start, k_stop, actual=None):
    """Return the cluster range
    
    Accepts the design matrix, the clustering object,
    the initial and final values to step through,
    and, optionally, actual labels. This will return a 
    dictionary of the cluster labels, cluster centers, 
    internal validation values and, if actual labels is
    given, external validation values, for every k.
    """
    
    ys = []
    centers = []
    inertias = []
    chs = []
    scs = []
    ps = []
    amis = []
    ars = []
    for k in range(k_start, k_stop+1):
        clusterer_k = clone(clusterer)
        # YOUR CODE HERE
        clusterer_k.set_params(n_clusters=k)
        clusterer_k.fit(X)
        #raise NotImplementedError() 
        y = clusterer_k.labels_ #label
        ys.append(y)
        centers.append(clusterer_k.cluster_centers_) #centering
        inertias.append(clusterer_k.inertia_) #inertia
        chs.append(calinski_harabasz_score(X,y))
        scs.append(silhouette_score(X,y))
       
        if actual is not None:
            ps.append(purity(actual, y))   
            amis.append(adjusted_mutual_info_score(actual, y))
            ars.append(adjusted_rand_score(actual, y)) 
    cluster_dict = {'ys': ys,
                    'centers': centers,
                    'inertias': inertias,
                    'chs': chs,
                    'scs': scs
                    }
        
    if actual is not None:
        cluster_dict['ps'] = ps
        cluster_dict['amis'] = amis
        cluster_dict['ars'] = ars
    return cluster_dict


def plot_internal(inertias, chs, scs,f_num=24):
    """Plot internal validation values"""
    
    fig, ax = plt.subplots(figsize=(8,5))
    ks = np.arange(2, len(inertias)+2)
    ax.plot(ks, inertias, '-o', label='SSE')
    ax.plot(ks, chs, '-ro', label='CH')
    ax.set_xlabel('$k$', fontsize=14)
    ax.set_ylabel('SSE/CH', fontsize=14)
    lines, labels = ax.get_legend_handles_labels()
    ax2 = ax.twinx()
    
    ax2.plot(ks, scs, '-ko', label='Silhouette coefficient')
    ax2.set_ylabel('Silhouette', fontsize=14)
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines+lines2, labels+labels2, bbox_to_anchor=(1.4,1))
    plt.title('Figure '+str(f_num)+'. Internal validation values for '
                 'different k', fontsize=16)
    plt.show()
    
def plot_external(ps, amis, ars, f_num=25):
    """Plot external validation values"""

    fig, ax = plt.subplots(figsize=(8,5))
    ks = np.arange(2, len(ps)+2)
    ax.plot(ks, ps, '-o', label='PS')
    ax.plot(ks, amis, '-ro', label='AMI')
    ax.plot(ks, ars, '-go', label='AR')
    ax.set_xlabel('$k$', fontsize=14)
    ax.set_ylabel('PS/AMI/AR', fontsize=14)
    ax.legend(bbox_to_anchor=(1.15,1))
    plt.title('Figure '+str(f_num)+'. External validation values for '
                 'different k', fontsize=16)
    plt.show()


def ward_cluster_search(X, f_num=26):
    """Implement ward's method on varying number of k clusters"""
    
    X = pd.DataFrame(X)

    fig, ax = plt.subplots(2, 5, dpi=150, sharex=True, sharey=True, 
                           figsize=(8,5),
                           subplot_kw=dict(aspect='equal'),
                           gridspec_kw=dict(wspace=0.01));
    for i in range(2, 12):
        agg = AgglomerativeClustering(n_clusters=i)
        y = agg.fit_predict(X)
        if i < 7:
            ax[0][i%7-2].scatter(X[0], X[1], s=12, c=y, alpha=0.5, 
                                 marker='.');
            ax[0][i%7-2].set_title('$k=%d$'%i);
        else:
            ax[1][i%7].scatter(X[0], X[1], s=12, c=y, alpha=0.5, marker='.');
            ax[1][i%7].set_title('$k=%d$'%i)
            
    plt.suptitle('Figure '+str(f_num)+". Ward's Hierarchical clustering "
              'for different k using the First and Second Singular Values',
             fontsize=16)
    plt.show()


def plot_dendrogram(X, f_num=27):
    """Return the hierarchical clustering encoded as a linkage matrix `Z` and
    plot the dendrogram."""
    
    Z = linkage(X, method='ward', optimal_ordering=True)
    fig, ax = plt.subplots(figsize=(12,4), dpi=300)
    dn = dendrogram(Z, ax=ax)
    ax.set_ylabel(r'$\Delta$', fontsize=14)
    plt.title("Figure "+str(f_num)+". Dendrogram using Ward's Agglomerative "
              "method", fontsize=16)
    return Z


def fancy_dendrogram( *args, **kwargs):
    """Plot the simpler or truncated version of the dendrogram."""
    
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)
    plt.figure(figsize=(8, 5))
    
    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
            
    plt.title("Figure "+str(28)+". Clustering Dendrogram (truncated)"
           , fontsize=16)
    plt.show()


def similar_color_func(word=None, font_size=None,
                       position=None, orientation=None,
                       font_path=None, random_state=None):
    """Return color funtion."""
    h = 40 # 0 - 360c
    s = 100 # 0 - 100
    l = random_state.randint(30, 70) # 0 - 100
    return "hsl({}, {}%, {}%)".format(h, s, l)


def plot_word_cloud(X_new, k_cluster=3, label_col='y_kmeans', model='Kmeans'):
    """Plot the common features in the cluster using
    Word cloud."""
    
    for k in range(0,k_cluster):

        df = X_new[X_new[label_col]==k]
        cols = df.columns
        if 'y_ward' in cols:
            df = df.drop(['y_ward'], axis=1)
        if 'y_kmeans' in cols:
            df = df.drop(['y_kmeans'], axis=1)
        if 'helpful_votes' in cols:
            df = df.drop(['helpful_votes'], axis=1)
            
        df = df[df.sum(axis=1)!=0]

        wordcloud = WordCloud(max_font_size=40, max_words=50, min_font_size=6,
                              background_color="#000B29", 
                            color_func=similar_color_func,
                           ).generate_from_frequencies(df.T.sum(axis=1))

        #show
        plt.figure(figsize=(10,10))
        plt.title('Cluster '+str(k)+' using '+model, fontsize=16)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()


def classifier_model(X, y, f_num):
    """Plot the Precision-Recall curve."""
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                       test_size = 0.2,
                                                        random_state = 42,
                                                        stratify = y)

    #Label Binarizer
    from sklearn.preprocessing import LabelBinarizer
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(y)
    n_classes = 3
    y_train_ = label_binarizer.transform(y_train)
    y_test_ = label_binarizer.transform(y_test)
    
    
    classes = [-1, 0, 1]
    classifier = OneVsRestClassifier(svm.LinearSVC(random_state=42))
    classifier.fit(X_train, y_train)
    y_score = classifier.decision_function(X_test)

    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i,j in enumerate(classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(y_test_[:, i], 
                                                       y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
                                      y_test_.ravel(),y_score.ravel())
    average_precision["micro"] = average_precision_score(y_test_, y_score,
                                                         average="micro")

    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue',
                    'teal'])

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))

    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        class_ = classes[i]
        labels.append('Precision-recall for class ' +str(class_)+":"+
                      str(round(average_precision[i],3)))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Figure '+str(f_num)+'. Extension of Precision-Recall curve '
              'to multi-class', fontsize=15)
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))


    plt.show()
    
    
    