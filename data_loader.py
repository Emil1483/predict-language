import random

import numpy as np

alphabet = "abcdefghijklmnopqrstuvwxyz"
max_letters = 15


def gen_nor_words():
    with open("raw_data/nor_conllu.txt", "r") as f:
        for line in f:
            word = line.split("\t")[2].lower()
            if all(letter in alphabet for letter in word):
                yield word


def gen_eng_words():
    with open("raw_data/eng_words.txt", "r") as f:
        for line in f:
            word = line.split("\n")[0].lower()
            if all(letter in alphabet for letter in word):
                yield word


def divide_list(l: list, bias):
    random.shuffle(l)
    n = round(len(l) * bias)
    return l[:n], l[n:]


def create_csvs():
    nor_words = [*gen_nor_words()]
    eng_words = [*gen_eng_words()]

    nor_words_train, nor_words_test = divide_list(nor_words, 0.99)
    eng_words_train, eng_words_test = divide_list(eng_words, 0.99)

    with open("data/train.csv", "w", newline="") as f:
        f.write("\n".join(w + ",0" for w in nor_words_train))
        f.write("\n")
        f.write("\n".join(w + ",1" for w in eng_words_train))

    with open("data/test.csv", "w", newline="") as f:
        f.write("\n".join(w + ",0" for w in nor_words_test))
        f.write("\n")
        f.write("\n".join(w + ",1" for w in eng_words_test))


def vectorized_word(word: str):
    if len(word) > max_letters:
        raise ValueError(f'word "{word}" too long')

    e = np.zeros((len(alphabet) * max_letters, 1))
    for i, letter in enumerate(word):
        index = alphabet.index(letter) + len(alphabet) * i
        e[index] = 1.0

    return e


def vectorized_result(j):
    e = np.zeros((2, 1))
    e[j] = 1.0
    return e


def gen_data(file: str):
    with open(file, "r") as f:
        for line in f:
            word, language = line.strip().split(",")

            if len(word) > max_letters:
                continue

            yield word, int(language)


def load_data():
    train_words, train_languages = list(zip(*gen_data("data/train.csv")))
    training_inputs = [vectorized_word(word) for word in train_words]
    training_results = [vectorized_result(y) for y in train_languages]
    training_data = list(zip(training_inputs, training_results))

    test_words, test_languages = list(zip(*gen_data("data/test.csv")))
    test_inputs = [vectorized_word(word) for word in test_words]
    test_data = list(zip(test_inputs, test_languages))

    return training_data, test_data


def get_multipliers():
    _, train_languages = list(zip(*gen_data("data/train.csv")))
    _, test_languages = list(zip(*gen_data("data/test.csv")))
    counts = {0: 0, 1: 0, 2: 0}
    for language in [*train_languages, *test_languages]:
        counts[language] += 1

    counts_max = max(counts.values())
    return [counts_max / c for c in counts.values()]


if __name__ == "__main__":
    create_csvs()

    training_data, _ = load_data()
    print("single training data sample:")
    print(training_data[0])
