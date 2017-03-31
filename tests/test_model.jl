using Knet
include("../util/infst.jl")
include("../util/chproces.jl")
include("../models/charfinal_model.jl")


function test_model()
    atype = (gpu() >= 0 ? KnetArray{Float32} : Array{Float32})

    # word model initialization
    ptb = open("../ptb/ptb.train.txt")
    sdict = Dict{Int64, Array{Any, 1}}(); ulimit=35; maxlines = 500; batchsize = 6;
    word_vocab = create_vocab("../ptb/ptb.vocab")
    readstream!(ptb, sdict, word_vocab;maxlines=1000, ulimit=ulimit)
    ids = nextbatch(ptb, sdict, word_vocab, batchsize; ulimit=ulimit, maxlis=maxlines)
    i2w = Array(AbstractString, length(word_vocab));
    for (k, v) in word_vocab; i2w[v] = k;end;

    # character settings
    ch1 = create_chvocab("../ptb/ptb.vocab")
    i2c = Array(Char, length(ch1));
    for (k, v) in ch1; i2c[v]=k ;end;

    # model initialization
    hiddens = [256]; charhidden = [256]; charvocab = length(ch1); wordvocab = length(word_vocab);
    m = initmodel(atype, hiddens, charhidden, charvocab, wordvocab);
    states = initstate(atype, hiddens, batchsize)
    schar = initstate(atype, charhidden, batchsize);
    
    println("the sequence length is $(length(ids))")
    # This part is only for testing the character based lstm
    # embeddings = Array(Any, length(ids))
    # for i=1:length(ids)
    #     embeddings[i] = charembed(m[:char], schar, ids[i], i2w, ch1, atype)
    # end

    # To make a gradcheck open the following 2 lines of code
    #gradcheck(charbilstm, m, schar, states, ids, i2w, ch1; gcheck=30, verbose=true, atol=0.01)
    #atype = Array{Float64}

    
    val = []
    lval = charbilstm(m, schar, states, ids, i2w, ch1, val)
    @show val
    @show lval
    ids = nextbatch(ptb, sdict, word_vocab, batchsize; ulimit=ulimit, maxlis=maxlines)
    lval = charbilstm(m, schar, states, ids, i2w, ch1, val)
    
    gs = gradcharbilstm(m, schar, states, ids, i2w, ch1, val)
    @show val
    
    
end
!isinteractive() && test_model()
