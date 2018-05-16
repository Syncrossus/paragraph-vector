from math import isnan, isinf, exp


def infNan(x):
    """ Raises an exception if x is inf or nan
    """
    if isnan(x) or isinf(x):
        raise Exception("x is inf or nan")


def cosDis(a, b):
    """ returns cosine similarity between a and b
    """
    return a.col(0).dot(b.col(0)) / (a.norm() * b.norm())


def sigmoid(x):
    return 1 / (1 + exp(-x))


def save(ofs, mat):
    """ saves mat (a matrix) to ofs (a file-like)
    """
    import pickle
    with open(ofs, "w") as f:
        pickle.dump(mat, f)


def load(ifs):
    """ retrieves and returns mat (a matrix) from ifs
        (a file-like in which mat was pickled)
    """
    import pickle
    return pickle.load(ifs)


# def procArg(wordVecDim, paragraphVecDim, contextSize,
#             learningRate, numNegative, minFreq, iteration,
#             numThreads, input_file, output):
def procArg(args):
    from sys import argv, exit

    try:
        # pairing arguments with their value
        argv_formatted = [(argv[i], argv[i + 1]) for i in range(1, len(argv), 2)]
    except IndexError:
        print("### Options ###\n\
        -wvdim    the dimensionality of word vectors (default: 50)\n\
        -pvdim    the dimensionality of paragraph vectors (default: 50)\n\
        -window   the context window size (default: 5)\n\
        -lr       the learning rate (default: 0.025)\n\
        -neg      the number of negative samples for negative sampling learning (default: 5)\n\
        -minfreq  the threshold to cut rare words (default: 10)\n\
        -itr      the number of iterations (default: 1)\n\
        -threads  the number of threads used (default: 1)\n\
        -input    the input file name (default: INPUT.txt)\n\
        -output   the output file name (default: OUTPUT)")
        exit(1)

    for option, value in argv_formatted:
        args[option] = value
