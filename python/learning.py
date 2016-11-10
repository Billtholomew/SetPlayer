import random
from scipy.cluster.vq import kmeans, whiten, vq
from numpy import array, linalg, log, max, min, argwhere


def classify_attributes(all_cards, attributes_to_classify):
    cards = all_cards.values()
    for attribute in attributes_to_classify:
        attribute_data = map(lambda card: card.attributes[attribute].data, all_cards.values())
        attribute_data = array(attribute_data)
        k, labels, centroids = learn_classes_kmeans(attribute_data)
        print attribute, k, labels
        map(lambda x: x[0].attributes[attribute].classify(x[1], centroids[x[1]]), zip(cards, labels))


def calculate_kmeans_data(white_data, k):
    centroids, distortion = kmeans(white_data, k)
    labels, _ = vq(white_data, centroids)
    wk = calculate_wk(centroids, labels, white_data)
    return {'wk': wk, 'labels': labels, 'centroids': centroids}


# Pass in a list of a the data for a single attribute for all cards
def learn_classes_kmeans(data):
    white_data = whiten(data)
    kmeans_data = get_kmeans_data(white_data)

    best_k = find_best_k(kmeans_data)

    best_labels = kmeans_data[best_k]['labels']
    best_centroids = kmeans_data[best_k]['centroids']

    return best_k, best_labels, best_centroids


# normalized intra-cluster sums of squares
# see here: https://datasciencelab.wordpress.com/2013/12/27/finding-the-k-in-k-means-clustering/
def calculate_wk(centroids, labels, data):
    clusters = {l: data[labels == l] for l in set(labels)}
    return log(sum([linalg.norm(centroids[i] - c) ** 2 / (2 * len(c)) for i, c in clusters.iteritems()]))


def generate_reference_data(k, size, feature_vector_bounds):
    reference_data = array(map(lambda w: map(lambda j:
                                             random.uniform(feature_vector_bounds[j][0], feature_vector_bounds[j][1]),
                                             xrange(len(feature_vector_bounds))),
                               xrange(size)))
    return calculate_kmeans_data(reference_data, k)


def get_kmeans_data(white_data):
    feature_vector_bounds = zip(min(white_data, axis=0), max(white_data, axis=0))
    # Dispersion for real distribution
    kmeans_data = {k + 1: calculate_kmeans_data(white_data, k + 1) for k in xrange(len(white_data))}
    ks = kmeans_data.keys()
    b = 12
    for k in ks:
        # Create B reference datasets
        b_wkbs = map(lambda b: generate_reference_data(k, len(white_data), feature_vector_bounds)['wk'], xrange(b))
        gap = (sum(b_wkbs) - kmeans_data[k]['wk']) / b
        sdk = (sum((b_wkbs - sum(b_wkbs) / b) ** 2) / b) ** 0.5
        sk = sdk * (1 + 1 / b) ** 0.5
        kmeans_data[k]['gap'] = gap
        kmeans_data[k]['sk'] = sk
    return kmeans_data

def find_best_k(kmeans_data):
    ks = sorted(kmeans_data.keys())
    gap = map(lambda k: kmeans_data[k]['gap'], ks)
    sk = map(lambda k: kmeans_data[k]['sk'], ks)
    gap_sk = map(lambda (g, s): g-s, zip(gap, sk))
    gap_diff = map(lambda (gk, gsk): gk - gsk, zip(gap, gap_sk[1:]))
    return min(argwhere(array(gap_diff) >= 0)) + 1


