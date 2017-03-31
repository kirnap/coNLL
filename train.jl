using Knet, ArgParse, JLD
include("util/infst.jl")
include("util/chproces.jl")
include("models/charfinal_model.jl")

function main(args=ARGS)
    s = ArgParseSettings()
    s.exc_handler = ArgParse.debug_handler
    @add_arg_table s begin
        ("--trainfile"; required=true; help="Infinite stream training file")
        ("--devfile"; required=true; help="dev data to test perplexity")
        ("--vocabfile"; required=true; help="Vocabulary file to train a model")
        ("--atype"; default=(gpu() >= 0 ? "KnetArray{Float32}" : "Array{Float32}"))
        ("--hiddens"; arg_type=Int; nargs='+'; default=[300]; help="hidden layer configuration")
        ("--embedding"; arg_type=Int; default=256)
        ("--batchsize"; arg_type=Int; default=25)
        ("--gclip"; arg_type=Float64; default=5.0)
        ("--epochs"; arg_type=Int; default=10)

    end
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    atype = eval(parse(o[:atype]))
    for (k, v) in o
        println("$k => $v")
    end
    

    
    # word data initialization
    ulimit = 35; maxlines = 500;
    word_vocab = create_vocab(o[:vocabfile])

    i2w = Array(AbstractString, length(word_vocab))
    for (k, v) in word_vocab; i2w[v] = k;end;

    
    # character vocabulary initialization
    char_vocab = create_chvocab(o[:vocabfile])

    # model initialization
    charhidden = [o[:embedding]]
    vsize = length(word_vocab); csize = length(char_vocab);
    m = initmodel(atype, o[:hiddens], charhidden, csize, vsize)
    sbil = initstate(atype, o[:hiddens], o[:batchsize])
    schar = initstate(atype, charhidden, o[:batchsize])
    opts = oparams(m, Adam; gclip=o[:gclip])

    dev = create_testdata(o[:devfile], word_vocab, 5)
    sbil_dev = initstate(atype, o[:hiddens], 5)
    schar_dev = initstate(atype, charhidden, 5)
    dperp = devperp(m, schar_dev, sbil_dev, dev, i2w, char_vocab)
    println("Initial devperp $dperp")
    flush(STDOUT)
    
    for i=1:o[:epochs]
        tloss = []
        (ptb, sdict) = create_data_environment(o[:trainfile], word_vocab; maxlines=1000, ulimit=ulimit)
        ids = nextbatch(ptb, sdict, word_vocab, o[:batchsize]; ulimit=ulimit, maxlines=maxlines)
        while ids !=nothing
            train(m, schar, sbil, ids, i2w, char_vocab, tloss, opts)
            ids = nextbatch(ptb, sdict, word_vocab, o[:batchsize]; ulimit=ulimit, maxlines=maxlines)
        end
        tperp = exp(mean(tloss))
        dperp = devperp(m, schar_dev, sbil_dev, dev, i2w, char_vocab)
        println("Epoch $i | dperp $dperp | trainperp $tperp")
        flush(STDOUT)
        close(ptb)
    end
end
!isinteractive() && main(ARGS)


# PTB INFO: total ptb.train.txt sequence numbers : 42068, available for training 39361
