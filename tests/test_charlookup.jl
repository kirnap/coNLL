include("../util/chproces.jl")
include("../util/infst.jl")

function test_charlookup()
    ulimit = 40; batchsize=5; maxlines = 500;

    word_vocab = create_vocab("../ptb/ptb.vocab");
    i2w = Array(AbstractString, length(word_vocab));
    for (k, v) in word_vocab; i2w[v] = k;end;
    (ptb, sdict) = create_data_environment("../ptb/ptb.train.txt", word_vocab; ulimit=ulimit, maxlines=1000)
    ch1 = create_chvocab("../ptb/ptb.vocab");
    i2c = Array(Char, length(ch1));
    for (k, v) in ch1; i2c[v]=k ;end;
    
    ids = nextbatch(ptb, sdict, word_vocab, batchsize; ulimit=ulimit, maxlines=maxlines);
    wids = ids[rand(1:length(ids))]
    data, _ = cbatchlookup(wids, i2w, ch1)

    # for sanity don't try to understand that part after first trial:)
    words = map(x->i2w[x], wids)
    for word in words; print("$word\n");end;println("---");
    padded_words = map(x->ibuild_word_lookup(i2c, data, x), collect(1:batchsize))
    for item in padded_words; for chx in item; print("$chx");end;println();end;

    #==#

end
!isinteractive() && test_charlookup()
