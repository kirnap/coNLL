include("../util/chproces.jl")
include("../util/infst.jl")

function test_char_mask_lookup()
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

    (data, masks) = cbatchlookup(wids, i2w, ch1)
    
    words = map(x->i2w[x], wids)
    for word in words;println("$word");end;println("---")
    pchs = Any[]
    for i=1:length(data)
        item = data[i]; mitem= masks[i];
        chars = Any[]
        for c in item; append!(chars, i2c[c]);end;
        ds = Any[]
        for k=1:length(chars)
            x = (chars[k], mitem[k])
            push!(ds, x)
        end
        push!(pchs, ds)
    end
    for i=1:length(pchs[1]); for item in pchs; print("$(item[i]) ");end;println();end;
    return (data, masks)
end
!isinteractive() && test_char_mask_lookup()
