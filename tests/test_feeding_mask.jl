# test masking process
include("../util/chproces.jl")
include("../util/infst.jl")


function get_desired_mask(data)
    bsize, chsize = size(data[1])
    masks = Any[]
    for i=1:length(data)
        s = data[i]
        m = ones(Float32, bsize, 1)
        for r=1:bsize
            tin = find(x->x==true, s[r, :])
            (tin == [1]) && (m[r] = 0)
        end
        push!(masks, m)
    end
    return masks
end


function test_mask()
    ptb = open("../ptb/ptb.train.txt")
    sdict = Dict{Int64, Array{Any, 1}}();
    ulimit = 40
    maxlines = 500
    batchsize = 6
    word_vocab = create_vocab("../ptb/ptb.vocab")
    readstream!(ptb, sdict, word_vocab;maxlines=1000, ulimit=ulimit)
    ids = nextbatch(ptb, sdict, word_vocab, batchsize; ulimit=ulimit, maxlis=maxlines)

    i2w = Array(AbstractString, length(word_vocab))
    for (k, v) in word_vocab; i2w[v] = k;end;

    # character settings
    ch1 = create_chvocab("../ptb/ptb.vocab")
    i2c = Array(Char, length(ch1));
    for (k, v) in ch1; i2c[v]=k ;end;

    wids = ids[rand(1:length(ids))]
    words = map(x->i2w[x], wids)
    for word in words; print("$word\n");end;println("---");
    (data, masks) = charbatch(wids, i2w, ch1)
    padded_words = map(x->ibuild_word(i2c, data, x), collect(1:batchsize))
    for item in padded_words
        for chx in item
            print("$chx")
        end
        print("\n")
    end
    desired_masks = get_desired_mask(data)
    @assert(desired_masks == masks)
    info("Mask tests passed")
    return (data, masks, i2c)
end
!isinteractive() && test_mask()
