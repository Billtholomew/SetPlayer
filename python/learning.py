from scipy.cluster.vq import kmeans, kmeans2, whiten
from numpy import array, linalg


def classify_attributes(all_cards, attributes_to_classify):
    cards = all_cards.values()
    for attribute in attributes_to_classify:
        attribute_data = map(lambda card: card.attributes[attribute].data, all_cards.values())
        attribute_data = array(attribute_data)
        labels = learn_classes_kmeans(attribute_data)
        map(lambda x: x[0].attributes[attribute].classify(x[1]), zip(cards, labels))


def Wk(centroids, labels, data):
    cluster_ids = set(labels.tolist())
    clusters = {l:data[labels == l] for l in set(labels)}
    s = 0
    for i,c in clusters.iteritems():
        s_t = linalg.norm(centroids[i] - c) ** 2 / (2 * len(c))
        s += s_t
    # sum([linalg.norm(centroids[i] - c) ** 2 / (2 * len(c)) for i in range(k) for c in clusters[i]])
    pass
    return s


# Pass in a list of a the data for a single attribute for all cards
def learn_classes_kmeans(data):
    labels = []
    temp_labels = [0 for _ in data]
    k = 1
    while k < len(data):
        k += 1
        labels = temp_labels
        centroids, temp_labels = kmeans2(data, k)
        wk = Wk(centroids, temp_labels, data)
        print wk
    return labels
