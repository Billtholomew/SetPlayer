from scipy.cluster.vq import kmeans, whiten, vq
from numpy import array, linalg, max, std, apply_along_axis


def classify_attributes(all_cards, attributes_to_classify, max_k=None, worker_pool=None):
    for attribute in attributes_to_classify:
        attribute_data = map(lambda card: card.attributes[attribute].data, all_cards)
        attribute_data = array(attribute_data)
        k, labels, centroids = learn_classes_kmeans(attribute_data, max_k, worker_pool)
        map(lambda x: x[0].attributes[attribute].classify(x[1], centroids[x[1]]), zip(all_cards, labels))


def calculate_kmeans_data(data_k):
    data, k = data_k
    # how can we make it work for kmeans2?
    white_data = whiten(data)
    centroids, distortion = kmeans(white_data, k)
    labels, _ = vq(white_data, centroids)
    wk = calculate_wk(centroids, labels, white_data)
    # centroids are "whitened" so make them dirty again
    centroids = centroids * std(data, axis=0)
    return [wk, labels, centroids]


# Pass in a list of a the data for a single attribute for all cards
def learn_classes_kmeans(data, max_k=None, worker_pool=None):
    max_k = len(data) if max_k is None else max_k

    data_k = map(lambda k: (data, k + 1), xrange(max_k))

    if worker_pool:
        kmeans_data = worker_pool.map(calculate_kmeans_data, data_k)
    else:
        kmeans_data = map(lambda dk: calculate_kmeans_data(dk), data_k)

    wks, _, _ = zip(*kmeans_data)
    wks = map(lambda wk: wk / max(wks), wks)

    # find first k where change in wk is less than 0.1
    best_k = map(lambda (a, b): a-b < 0.1, zip(wks, wks[1:])).index(True)

    _, best_labels, best_centroids = kmeans_data[best_k]

    return best_k + 1, best_labels, best_centroids


# normalized intra-cluster sums of squares
# see here: https://datasciencelab.wordpress.com/2013/12/27/finding-the-k-in-k-means-clustering/
def calculate_wk(centroids, labels, data):
    clusters = {l: data[labels == l] for l in set(labels)}
    return sum([linalg.norm(centroids[i] - c) ** 2 / (2 * len(c)) for i, c in clusters.iteritems()])
