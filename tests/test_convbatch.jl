include("../util/chproces.jl")
include("../util/infst.jl")

function test_convbatch()
    ulimit = 40; batchsize=5; maxlines = 500;

    word_vocab = create_vocab("../ptb/ptb.vocab");
    i2w = Array(AbstractString, length(word_vocab));
    for (k, v) in word_vocab; i2w[v] = k;end;
    (ptb, sdict) = create_data_environment("../ptb/ptb.train.txt", word_vocab; ulimit=ulimit, maxlines=1000)
    ch1 = create_chvocab("../ptb/ptb.vocab");
    i2c = Array(Char, length(ch1));
    for (k, v) in ch1; i2c[v]=k ;end;

    
    ids = nextbatch(ptb, sdict, word_vocab, batchsize; ulimit=ulimit, maxlines=maxlines)

    tokens = ids[rand(1:length(ids))]
    # show which words are chosen
    words = map(x->i2w[x], tokens); for word in words; print("$word\n");end;println("\n---");
    
    
    @time cbatch = cbatch4conv(tokens, i2w, ch1)

    rev_words = map(item->map(x->i2c[x], item), cbatch)
    for item in rev_words; for k in item;print("$k");end;print("\n");end;println()
    
    



end
!isinteractive() && test_convbatch()
