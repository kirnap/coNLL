# to test charbatching feature of that library
include("../../util/infst2.jl")


function test_infst2char()
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
    ch1 = create_chvocab("../../ptb/ptb_all_vocab")
    i2c = Array(Char, length(ch1))
    for (k, v) in ch1; i2c[v] = k; end;
    wids = ids[rand(1:length(ids))]
    ch2lstm = charlup(wids, i2w_all, ch1)
    return (ids, i2w_all, i2w_out, ch2lstm, i2c)
end
