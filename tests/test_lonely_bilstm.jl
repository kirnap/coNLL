using Knet
include("../util/infst.jl")
include("../models/lonely_bilstm.jl")

function test_lonely_bilstm()
    atype = (gpu() >= 0 ? KnetArray{Float32} : Array{Float32})

    # word model initialization
    ulimit=35; maxlines = 500; batchsize = 20;; 
    
    word_vocab = create_vocab("../ptb/ptb.vocab")
    


    # model initialization
    hiddens = [300]; vocabsize = length(word_vocab); embedding=256;
    m = initmodel(atype, hiddens, embedding, vocabsize)
    s = initstate(atype, hiddens, batchsize)
    opts = oparams(m, Adam; gclip=5.0)


    # prepare test data
    dev = create_testdata("../ptb/ptb.valid.txt", word_vocab, 5)
    sdev = initstate(atype, hiddens, 5)

    dperp = devperp(m, sdev, dev)
    println("Initial dev perplexity is $dperp")

    counter = 10
    for i=1:counter
        tloss = []
        ptb, sdict = create_data_environment("../ptb/ptb.train.txt", word_vocab; ulimit=ulimit, maxlines=1000)
        ids = nextbatch(ptb, sdict, word_vocab, batchsize; ulimit=ulimit, maxlines=maxlines)
        while (ids != nothing)
            train(m, s, ids, opts, tloss)
            ids = nextbatch(ptb, sdict, word_vocab, batchsize; ulimit=ulimit, maxlines=maxlines)
        end
        tperp = exp(mean(tloss))
        dperp = devperp(m, sdev, dev)
        println("Epoch $i | dperp $dperp | trainperp $tperp")
        close(ptb)
    end

end

!isinteractive() && test_lonely_bilstm()
