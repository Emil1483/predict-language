from network import load_net, predict_language, Network


if __name__ == "__main__":
    net: Network = load_net()
    while True:
        word = input("type a word: ")
        predict_language(net, word)
