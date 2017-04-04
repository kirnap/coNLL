# create character tests
include("../chproces.jl")
include("../infst.jl")


function test_characters()

    print_single_sentence = false # open that to see how characters are build
    
    ptb = open("../ptb/ptb.train.txt")
    sdict = Dict{Int64, Array{Any, 1}}();
    ulimit = 40
    maxlines = 500
    batchsize = 3
    word_vocab = create_vocab("../ptb/ptb.vocab")
    readstream!(ptb, sdict, word_vocab;maxlines=1000, ulimit=ulimit)
    ids = nextbatch(ptb, sdict, word_vocab, batchsize; ulimit=ulimit, maxlis=maxlines)
    

    i2w = Array(AbstractString, length(word_vocab))
    for (k, v) in word_vocab; i2w[v] = k;end;

    # character settings
    ch1 = create_chvocab("../ptb/ptb.vocab")
    ch2 = create_chvocab("../ptb/ptb.train.txt")
    @assert(length(ch1) == length(ch2))
    for (k, v) in ch1; @assert(k in keys(ch2));end;
    info("character feeding test pass")

    i2c = Array(Char, length(ch1))
    for (k, v) in ch1; i2c[v]=k ;end;
    if print_single_sentence
        for word in ids
            chs = w2cs(word[1], i2w, ch1)
            recovered = ibuild_word(chs, i2c)
            for item in recovered; print(item);end;print(" ");
        end
        println()
    end
    return (ids, i2w, ch1)
    
end
!isinteractive() && test_characters()
