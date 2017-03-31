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
    ulimit = 40
    maxlines = 500
    word_vocab = create_vocab("../ptb/ptb.vocab")
    (ptb, sdict) = create_data_environment("../ptb/ptb.train.txt", word_vocab; ulimit=ulimit, maxlines=1000)
        

    counter = 0
    ids = nextbatch(ptb, sdict, word_vocab, batchsize; ulimit=ulimit, maxlines=maxlines)
    while ids != nothing
        ids = nextbatch(ptb, sdict, word_vocab, batchsize; ulimit=ulimit, maxlines=maxlines)
        counter += 1
    end
    info("nothing tests pass with $counter many lines")

    for i=1:5
        (ptb, sdict) = create_data_environment("../ptb/ptb.train.txt", word_vocab; ulimit=ulimit, maxlines=1000)
        ids = nextbatch(ptb, sdict, word_vocab, batchsize; ulimit=ulimit, maxlines=maxlines)
        counter = 0
        while ids != nothing
            ids = nextbatch(ptb, sdict, word_vocab, batchsize; ulimit=ulimit, maxlines=maxlines)
            counter += 1
        end
        close(ptb)
        info("nothing tests pass with $counter many lines in the $(i)th iteration")
    end

    return (ids, sdict)

end
!isinteractive() && testfeed()
