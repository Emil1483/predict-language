import gzip
import pickle
from data_loader import load_data
from network import Network


if __name__ == "__main__":
    print("loading data...")
    training_data, test_data = load_data()
    print("learning...")
    net = Network([390, 30, 2])
    net.SGD(training_data, 5, 50, 3.0, test_data=test_data)

    print("saving network...")
    with gzip.open("net.pkl.gz", "wb") as f:
        pickle.dump(net, f)
