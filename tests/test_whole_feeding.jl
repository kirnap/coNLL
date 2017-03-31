using Knet
include("../util/infst.jl")
include("../util/chproces.jl")
include("../models/model.jl")

function test_whole_feeding()
    atype = (gpu() >= 0 ? KnetArray{Float32} : Array{Float32})

    # word model initialization
    ptb = open("../ptb/ptb.train.txt")
    sdict = Dict{Int64, Array{Any, 1}}(); ulimit=40; maxlines = 500; batchsize = 1;
    word_vocab = create_vocab("../ptb/ptb.vocab")
    readstream!(ptb, sdict, word_vocab;maxlines=1000, ulimit=ulimit)
    ids = nextbatch(ptb, sdict, word_vocab, batchsize; ulimit=ulimit, maxlis=maxlines)
    i2w = Array(AbstractString, length(word_vocab));
    for (k, v) in word_vocab; i2w[v] = k;end;

    # character settings
    ch1 = create_chvocab("../ptb/ptb.vocab")
    i2c = Array(Char, length(ch1));
    for (k, v) in ch1; i2c[v]=k ;end;

    # model parameters initialization
    hiddens = [256]; charhidden = [256]; charvocab = length(ch1); wordvocab = length(word_vocab);
    m = initmodel(atype, hiddens, charhidden, charvocab, wordvocab);
    schar = initstate(atype, charhidden, batchsize);

    # Here with that line I test the single token feeding
    #(o1, input_to_lstm) = charbilstm(m, schar, schar, ids, i2w, ch1)

    # Here imagine taking single character and returning the hidden vector
    # ids[i] is the ith token of given sentences,
    # for ith token you need to have batchsize x hidden embeddings!
    (data, masks) = charbatch(ids[end], i2w, ch1)
    hs = []
    h = zeros(similar(schar[1]))
    mtot = atype(zeros(similar(masks[1])))
    # Here you are moving a single word forward
    iteration=1;
    for (ch, mo) in zip(data, masks);
        @show iteration; iteration +=1
        cbon = convert(atype, ch)
        mbon = convert(atype, mo)
        h1 = chforw(m[:char], schar, cbon; mask=mbon)
        mtot += mbon
        h = h1 # this line is for getting the final hidden state of the lstm
        #h += h1 # this line is for averaging
        push!(hs, h1) # The last item of hs is the final hidden state of the lstm
    end
    return (hs, h, mtot)
end
!isinteractive() && test_whole_feeding()

