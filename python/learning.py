from scipy.cluster.vq import kmeans, kmeans2, whiten, vq
from numpy import array, linalg, mean, std


def classify_attributes(all_cards, attributes_to_classify):
    cards = all_cards.values()
    for attribute in attributes_to_classify:
        attribute_data = map(lambda card: card.attributes[attribute].data, all_cards.values())
        attribute_data = array(attribute_data)
        labels, centroids = learn_classes_kmeans(attribute_data)
        print attribute, labels
        map(lambda x: x[0].attributes[attribute].classify(x[1], centroids[x[1]]), zip(cards, labels))


# normalized intra-cluster sums of squares
# see here: https://datasciencelab.wordpress.com/2013/12/27/finding-the-k-in-k-means-clustering/
def calculate_wk(centroids, labels, data):
    clusters = {l: data[labels == l] for l in set(labels)}
    return sum([linalg.norm(centroids[i] - c) ** 2 / (2 * len(c)) for i, c in clusters.iteritems()])


# Pass in a list of a the data for a single attribute for all cards
def learn_classes_kmeans(data):
    print '--------'
    #for d in data:
    #    print ','.join(map(str,d))
    white_data = whiten(data)
    k = 0
    centroids = [mean(data)]
    labels = [0 for _ in data]
    prev_wk = float('inf')
    wk = calculate_wk(centroids, labels, white_data) * len(data) # basically just a really big number
    while abs(prev_wk - wk) > 0.075 and k < len(data):
        k += 1
        prev_wk = wk
        best_centroids = centroids
        best_labels = labels
        centroids, distortion = kmeans(white_data, k)
        labels, _ = vq(white_data, centroids)
        wk = calculate_wk(centroids, labels, white_data)
        print k, wk, distortion
        # cost for adding another class. 1.5 or greater found experimentally to work
    #labels, centroids = vq(white_data, best_centroids)

    best_centroids *= std(data, axis=0)

    return best_labels, best_centroids
