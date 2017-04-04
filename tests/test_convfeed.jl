# timing and feeding test
using Knet
include("../util/chproces.jl")
include("../util/infst.jl")
function test_convfeed()
    atype = gpu() >= 0 ? KnetArray{Float32} : Array{Float32}
    ulimit = 40; batchsize=40; maxlines = 500;

    word_vocab = create_vocab("../ptb/ptb.vocab");
    i2w = Array(AbstractString, length(word_vocab));
    for (k, v) in word_vocab; i2w[v] = k;end;
    (ptb, sdict) = create_data_environment("../ptb/ptb.train.txt", word_vocab; ulimit=ulimit, maxlines=1000)
    ch1 = create_chvocab("../ptb/ptb.vocab");
    i2c = Array(Char, length(ch1));
    for (k, v) in ch1; i2c[v]=k ;end;
    
    ids = nextbatch(ptb, sdict, word_vocab, batchsize; ulimit=ulimit, maxlines=maxlines);
    wids = ids[rand(1:length(ids))]
    cbatch = cbatch4conv(wids, i2w, ch1)

    l=longest_word(word_vocab)+2 # +2 for start and end tokens
    d=5; w=2; filter_bank = 2;
    H = convert(atype, randn(w, d, 1, filter_bank)); bias = convert(atype, zeros(1, 1, filter_bank, 1))
    
    chsize=length(ch1);
    chembed = convert(atype, randn(chsize, d));
    bm = nothing
    while ids != nothing
        for wids in ids
            cbatch = cbatch4conv(wids, i2w, ch1)
            ewem = []
            for word in cbatch
                x1 = chembed[word, :]
                if length(word) < l
                    (rows, cols) = l - length(word),d
                    pad = atype(zeros(rows, cols))
                    x1 = vcat(x1, pad)
                end
                push!(ewem, x1)
            end
            try
                bm = hcat(ewem...)
            catch
                warn("some words are longer than expected")
                if isinteractive()
                    for word in cbatch
                        if length(word) > l
                            return word
                        end
                    end
                end
            end
        end
        ids = nextbatch(ptb, sdict, word_vocab, batchsize; ulimit=ulimit, maxlines=maxlines);
    end
    c_k = reshape(bm, (l, d, 1, batchsize))
    # paramaters
    filter_bank = 20; pwin = l + w - 1; bias = atype(zeros(1,1, filter_bank, 1));

    # convolution operation
    H = atype(randn(w, d, 1, filter_bank))
    y_k = pool(tanh(conv4(H, c_k) .+ bias); window=pwin)
    fin = mat(y_k)' # this is a single time step input to bilstm
    @assert(size(fin)==(batchsize, filter_bank))
    info("Size test of the batched tokens passed")
    #return (cbatch, i2c, bm, c_k, ewem, y_k, fin)
end
!isinteractive() && test_convfeed()
