from scipy.cluster.vq import kmeans, kmeans2, whiten
from numpy import array


def classify_attributes(all_cards, attributes_to_classify):
    cards = all_cards.values()
    for attribute in attributes_to_classify:
        attribute_data = map(lambda card: card.attributes[attribute].data, all_cards.values())
        attribute_data = array(attribute_data)
        labels = learn_classes_kmeans(attribute_data)
        map(lambda x: x[0].attributes[attribute].classify(x[1]), zip(cards, labels))


# Pass in a list of a the data for a single attribute for all cards
def learn_classes_kmeans(data):
    labels = []
    temp_labels = [0 for _ in data]
    k = 1
    while len(set(temp_labels)) == k:
        k += 1
        labels = temp_labels
        _, temp_labels = kmeans2(data, k)
    return labels
