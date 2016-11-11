from scipy.cluster.vq import kmeans, whiten, vq
from numpy import array, linalg, max, std


def classify_attributes(all_cards, attributes_to_classify, max_k=None):
    cards = all_cards.values()
    for attribute in attributes_to_classify:
        attribute_data = map(lambda card: card.attributes[attribute].data, all_cards.values())
        attribute_data = array(attribute_data)
        k, labels, centroids = learn_classes_kmeans(attribute_data, max_k)
        map(lambda x: x[0].attributes[attribute].classify(x[1], centroids[x[1]]), zip(cards, labels))


def calculate_kmeans_data(data, k):
    # how can we make it work for kmeans2?
    white_data = whiten(data)
    centroids, distortion = kmeans(white_data, k)
    labels, _ = vq(white_data, centroids)
    wk = calculate_wk(centroids, labels, white_data)
    # centroids are "whitened" so make them dirty again
    centroids = centroids * std(data, axis=0)
    return {'wk': wk, 'labels': labels, 'centroids': centroids}


# Pass in a list of a the data for a single attribute for all cards
def learn_classes_kmeans(data, max_k=None):
    max_k = len(data) if max_k is None else max_k
    kmeans_data = map(lambda ik: calculate_kmeans_data(data, ik + 1), xrange(max_k))

    ks = range(max_k)
    wks = map(lambda ik: kmeans_data[ik]['wk'], ks)
    wks = map(lambda ik: wks[ik] / max(wks), ks)

    k = 0
    best_k = k
    diff = 1
    while k < max_k and diff >= 0.1:
        best_k = k
        k += 1
        diff = wks[k-1] - wks[k]

    best_labels = kmeans_data[best_k]['labels']
    best_centroids = kmeans_data[best_k]['centroids']

    return best_k + 1, best_labels, best_centroids


# normalized intra-cluster sums of squares
# see here: https://datasciencelab.wordpress.com/2013/12/27/finding-the-k-in-k-means-clustering/
def calculate_wk(centroids, labels, data):
    clusters = {l: data[labels == l] for l in set(labels)}
    return sum([linalg.norm(centroids[i] - c) ** 2 / (2 * len(c)) for i, c in clusters.iteritems()])
