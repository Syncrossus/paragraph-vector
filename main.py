import vocabulary
import utils


def main():
    settings = {
        "-wvdim": 50,
        "-pvdim": 50,
        "-window": 5,
        "-lr": 0.025,
        "-neg": 5,
        "-minfreq": 10,
        "-itr": 1,
        "-threads": 1,
        "-input": "INPUT.txt",
        "-output": "OUTPUT"
    }
    wordVecDim = settings["-wvdim"]
    paragraphVecDim = settings["-pvdim"]
    contextSize = settings["-window"]
    learningRate = settings["-lr"]
    numNegative = settings["-neg"]
    minFreq = settings["-minfreq"]
    iteration = settings["-itr"]
    numThreads = settings["-threads"]
    input_file = settings["-input"]
    output = settings["-output"]
    shrink = 0.0

    utils.procArg(settings)

    voc = vocabulary.Vocabulary(wordVecDim, contextSize, paragraphVecDim)

    voc.read(input_file, minFreq)
    shrink = learningRate / iteration

    for i in range(iteration):
        print("Iteration ", i + 1, " (current learning rate: ", learningRate, ")")
        voc.train(input_file, learningRate, shrink, numNegative, numThreads)
        learningRate -= shrink

    voc.save(output + ".bin")
    # voc.wordKnn(10)
    voc.outputParagraphVector(output + ".pv")
    voc.outputWordVector(output + ".wv")
