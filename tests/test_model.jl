using Knet
include("../util/infst.jl")
include("../util/chproces.jl")
include("../models/charfinal_model.jl")


function test_model()
    atype = (gpu() >= 0 ? KnetArray{Float32} : Array{Float32})

    # word model initialization
    ulimit=35; maxlines = 500; batchsize = 20;
    
    word_vocab = create_vocab("../ptb/ptb.vocab")
    ptb, sdict = create_data_environment("../ptb/ptb.train.txt", word_vocab; ulimit=ulimit, maxlines=1000)
    
    ids = nextbatch(ptb, sdict, word_vocab, batchsize; ulimit=ulimit, maxlis=maxlines)
    i2w = Array(AbstractString, length(word_vocab));
    for (k, v) in word_vocab; i2w[v] = k;end;

    # character settings
    ch1 = create_chvocab("../ptb/ptb.vocab")
    i2c = Array(Char, length(ch1));
    for (k, v) in ch1; i2c[v]=k ;end;

    # model initialization
    hiddens = [512]; charhidden = [256]; charvocab = length(ch1); wordvocab = length(word_vocab);
    m = initmodel(atype, hiddens, charhidden, charvocab, wordvocab);
    states = initstate(atype, hiddens, batchsize)
    schar = initstate(atype, charhidden, batchsize);
    opts = oparams(m, Adam; gclip=5.0)
    
    println("the sequence length is $(length(ids))")
    # This part is only for testing the character based lstm
    # embeddings = Array(Any, length(ids))
    # for i=1:length(ids)
    #     embeddings[i] = charembed(m[:char], schar, ids[i], i2w, ch1, atype)
    # end

    # To make a gradcheck open the following 2 lines of code
    #gradcheck(charbilstm, m, schar, states, ids, i2w, ch1; gcheck=30, verbose=true, atol=0.01)
    #atype = Array{Float64}

    

    dev = create_testdata("../ptb/ptb.valid.txt", word_vocab, 5)
    sdev = initstate(atype, hiddens, 5)
    scdev = initstate(atype, charhidden, 5)

    dperp = devperp(m, scdev, sdev, dev, i2w, ch1)
    @show dperp
    
    counter = 0
    val = []
    while ids != nothing
        train(m, schar, states, ids, i2w, ch1, val, opts)
        ids = nextbatch(ptb, sdict, word_vocab, batchsize; ulimit=ulimit, maxlines=maxlines)
        counter += 1
    end
    @show exp(mean(val))
    
    
    dperp = devperp(m, scdev, sdev, dev, i2w, ch1)
    @show dperp
    
    
    
    
end
!isinteractive() && test_model()
