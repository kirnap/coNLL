# test file for infst2.jl convolutional batcher -> cbatch4conv
using Knet
include("../util/infst2.jl")

function test_conv4batch()
    atype = gpu() >= 0 ? KnetArray{Float32} : Array{Float32}
    atype = KnetArray{Float64}
    # word level initialization
    ptb = open("../ptb/ptb.train.txt")
    all_vocab = create_vocab("../ptb/ptb_all_vocab");
    out_vocab = create_vocab("../ptb/ptb2k.vocab");
    i2w_all = Array(AbstractString, length(all_vocab));
    i2w_out = Array(AbstractString, length(out_vocab));
    for (k, v) in out_vocab; i2w_out[v] = k; end;
    for (k, v) in all_vocab; i2w_all[v] = k; end;
    lw = longest_word(all_vocab) + 2 # plus 2 for SOW and EOW tokens
    

    # convolutional level initilaziation
    conv_vocab = create_conv_vocab(all_vocab)
    lw2 = longest_word(conv_vocab)
    @assert(lw2 == lw, "vocabulary is not counted good")
    i2w_all_conv = Array(AbstractString, length(conv_vocab));
    for (k, v) in conv_vocab; i2w_all_conv[v] = k; end;

    
    # character level initialization
    ch1 = create_chvocab("../ptb/ptb_all_vocab")
    i2c = Array(Char, length(ch1))
    for (k, v) in ch1; i2c[v] = k; end;

    # model parameters initialization
    hiddens = [300];
    windowlen = 5; # corresponds to w in the paper
    filterbank =20; # correponds to h in the paper
    chembedding=15; # corresponds to d in the paper
    charvocab = length(ch1) # character lengths
    outvocab = length(out_vocab); # output vocabulary length
    pwind = lw + chembedding - 1 # pooling window size

    # get random data
    maxlines = 100; s = Dict{Int64, Array{Any, 1}}();  ulimit=28; batchsize = 2;
    readstream!(ptb, s, out_vocab, all_vocab; maxlines=500, ulimit=28)
    ids = nextbatch(ptb, s, out_vocab, all_vocab, batchsize; maxlines=maxlines, ulimit=ulimit)
    windex = rand(1:length(ids))
    wids = ids[windex]

    word_ids = map(x->x[2], wids)
    words = map(x->i2w_all[x], word_ids)
    padded_words = map(x->i2w_all_conv[x[2]], wids)
    for word in words;println(word);end;println("---")
    for word in padded_words;println(word);end;println("---")

    (data, mask) = cbatch4conv(wids, i2w_all_conv, ch1, atype)
    words = []
    for i=1:lw:length(data)
        win = data[i:(i+lw-1)]
        push!(words, win)
    end
    info("These are the input for convolutional layer")
    for word in words
        x = map(x->i2c[x], word)
        for item in x;print(item);end;println()
    end
    return (words, i2c)
end
!isinteractive() && test_conv4batch()
