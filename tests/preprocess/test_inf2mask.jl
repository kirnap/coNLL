# test masking with 2nd version of the infinte streamer
include("../../util/infst2.jl")


function test_inf2mask()
    ptb = open("../../ptb/ptb.train.txt")
    all_vocab = create_vocab("../../ptb/ptb_all_vocab");
    out_vocab = create_vocab("../../ptb/ptb2k.vocab");
    i2w_all = Array(AbstractString, length(all_vocab));
    i2w_out = Array(AbstractString, length(out_vocab));
    for (k, v) in out_vocab; i2w_out[v] = k; end;
    for (k, v) in all_vocab; i2w_all[v] = k; end;


    maxlines = 1000; s = Dict{Int64, Array{Any, 1}}();  ulimit=100; batchsize = 5;

    ids = nextbatch(ptb, s, out_vocab, all_vocab, batchsize; maxlines=100)

    # get random batch of words
    ch1 = create_chvocab(all_vocab) #("../../ptb/ptb_all_vocab")
    i2c = Array(Char, length(ch1))
    for (k, v) in ch1; i2c[v] = k; end;
    wids = ids[rand(1:length(ids))]
    # in order to make sure about the changes in charlup2 you need to play with line 25
    (data, masks) = charlup2(wids, i2w_all, ch1) # this is input for lstm character level, data needs to be the real version of the words

    real_words = map(x->i2w_all[x[2]], wids)
    unked_words = map(x->i2w_out[x[1]], wids)
    for item in unked_words; print("$item\n");end;println("---> to charlstm--->");
    for item in real_words; print("$item\n");end;println("---");

    pchs = Any[]
    for i=1:length(data)
        item = data[i]; mitem= masks[i];
        chars = Any[]
        for c in item;append!(chars, i2c[c]);end;
        ds = Any[]
        for k=1:length(chars)
            x = (chars[k], mitem[k])
            push!(ds, x)
        end
        push!(pchs, ds)
    end
    for i=1:length(pchs[1]); for item in pchs; print("$(item[i]) ");end;println();end;
    
    (data2, masks2) = charlup3(wids, i2w_all, ch1)
    pchs2 = Any[]
    for i=1:length(data2)
        item = data2[i]; mitem = masks2[i];
        chars = Any[]
        for c in item;append!(chars, i2c[c]);end;
        ds = Any[]
        for k=1:length(chars)
            x = (chars[k], mitem[k]) # pm for put mask
            push!(ds, x)
        end
        push!(pchs2, ds)
    end
    println("\n3rd version of charlup-->")
    for i=1:length(pchs2[1]); for item in pchs2; print("$(item[i]) ");end;println();end;

    return (data2, i2w_all, ch1)

end
!isinteractive() && test_inf2mask()
