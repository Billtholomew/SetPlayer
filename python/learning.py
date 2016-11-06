from scipy.cluster.vq import kmeans, kmeans2, whiten, vq
from numpy import array, linalg


def classify_attributes(all_cards, attributes_to_classify):
    cards = all_cards.values()
    for attribute in attributes_to_classify:
        attribute_data = map(lambda card: card.attributes[attribute].data, all_cards.values())
        attribute_data = array(attribute_data)
        labels = learn_classes_kmeans(attribute_data)
        print attribute, labels
        map(lambda x: x[0].attributes[attribute].classify(x[1]), zip(cards, labels))


# Pass in a list of a the data for a single attribute for all cards
def learn_classes_kmeans(data):
    labels = []
    temp_labels = [0 for _ in data]
    k = 0
    print '--------'
    for d in data:
        print ','.join(map(str,d))
    best_distortion = float('inf')
    best_centroids = []
    white_data = whiten(data)
    while k < len(data):
        k += 1
        centroids, distortion = kmeans(white_data, k)
        # cost for adding another class. 1.5 or greater found experimentally to work
        distortion *= (2 ** k)
        print k, distortion
        if distortion < best_distortion:
            best_distortion = distortion
            best_centroids = centroids
    labels, centroids = vq(white_data, best_centroids)
    return labels
