# tests the token feeding to model
using Knet
include("../util/infst.jl")


function testfeed()
    atype = (gpu() >= 0 ? KnetArray{Float32} : Array{Float32})

    # hyper parameters
    batchsize = 1
    hiddens = [128]
    embedding = 128
    

    # prepare train data
    ptb = open("../ptb/ptb.train.txt")
    sdict = Dict{Int64, Array{Any, 1}}();
    ulimit = 40
    maxlines = 500
    word_vocab = create_vocab("../ptb/ptb.vocab")
    readstream!(ptb, sdict, word_vocab;maxlines=1000, ulimit=ulimit)
    ids = nextbatch(ptb, sdict, word_vocab, batchsize; ulimit=ulimit, maxlines=maxlines)

    index_to_word = Array(AbstractString, length(word_vocab))
    for (k, v) in word_vocab; index_to_word[v] = k; end;

    counter = 0
    while ids != nothing
        ids = nextbatch(ptb, sdict, word_vocab, batchsize; ulimit=ulimit, maxlines=maxlines)
        counter += 1
    end
    info("nothing tests pass with $counter many lines")

    return ids

end
!isinteractive() && testfeed()
