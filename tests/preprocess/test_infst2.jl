# Infinite streamer tests 2
include("../../util/infst2.jl")

function old_readstream!(f::IO,
                     s::Dict{Int64, Array{Any, 1}},
                     vocab::Dict{AbstractString, Int64},
                     realvocab::Dict{AbstractString, Int64};
                     maxlines=100, llimit=3, ulimit=300)
    
    k = 0
    while (k != maxlines)
        eof(f) && return false
        words = split(readline(f))
        ((length(words) < llimit) || (length(words) > ulimit-2)) && continue
        seq = Int32[]
        push!(seq, vocab[SOS])
        for word in words
            index = get(vocab, word, vocab[UNK])
            realindex = get(realvocab, word, realvocab[UNK]) # no unking activity desired
            push!(seq, index)
        end
        push!(seq, vocab[EOS])
        skey = length(seq)
        (!haskey(s, skey)) && (s[skey] = Any[])
        push!(s[skey], seq)
        k += 1
    end
end    


function test_infst()
    #real_vocabfile = "../../data/English/all_eng_vocab"
    #vocabfile = "../../data/English/all_eng12k.vocab"
    #real_vocab = create_realvocab(real_vocabfile)
    #vocab = create_vocab(vocabfile)

    #= to test with fake data
    batchsize=2;
    realvocab = create_vocab("fake_allvocab")
    fkvocab =  create_vocab("fake_5allvocab")
    i2w_real = Array(AbstractString, length(realvocab));
    for (k, v) in realvocab; i2w_real[v] = k; end;
    i2w = Array(AbstractString, length(fkvocab));
    for (k, v) in fkvocab; i2w[v] = k ;end;
    =#


    # ptb environment
    ptb = open("../../ptb/ptb.train.txt")
    all_vocab = create_vocab("../../ptb/ptb_all_vocab");
    out_vocab = create_vocab("../../ptb/ptb2k.vocab");
    i2w_all = Array(AbstractString, length(all_vocab));
    i2w_out = Array(AbstractString, length(out_vocab));
    for (k, v) in out_vocab; i2w_out[v] = k; end;
    for (k, v) in all_vocab; i2w_all[v] = k; end;

    # create readstream environment
    maxlines = 1000; s = Dict{Int64, Array{Any, 1}}();  ulimit=100; batchsize = 5;
    readstream!(ptb, s, out_vocab, all_vocab; maxlines=maxlines)
    ids = nextbatch(ptb, s, out_vocab, all_vocab, batchsize; maxlines=100)

    #old_readstream!(f, s, fkvocab, realvocab; maxlines=maxlines)
    s_before = deepcopy(s)


    return s, s_before, ids, i2w_out, i2w_all
end

