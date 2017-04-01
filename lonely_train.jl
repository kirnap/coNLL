using Knet, ArgParse, JLD
include("util/infst.jl")
include("models/lonely_bilstm.jl")

function main(args=ARGS)
    s = ArgParseSettings()
    s.exc_handler = ArgParse.debug_handler
    @add_arg_table s begin
        ("--trainfile"; required=true; help="Infinite stream training file")
        ("--devfile"; required=true; help="dev data to test perplexity")
        ("--vocabfile"; required=true; help="Vocabulary file to train a model")
        ("--savefile"; help="To save the julia model")
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

    # model initialization
    vsize = length(word_vocab)
    m = initmodel(atype, o[:hiddens], o[:embedding], vsize)
    s = initstate(atype, o[:hiddens], o[:batchsize])
    opts = oparams(m, Adam; gclip=5.0)

    # prepare test data
    dev = create_testdata(o[:devfile], word_vocab, 5)
    sdev = initstate(atype, o[:hiddens], 5)
    dperp = devperp(m, sdev, dev)
    println("Initial dev perplexity is $dperp")
    flush(STDOUT)

    for i=1:o[:epochs]
        tloss = []
        (ptb, sdict) = create_data_environment(o[:trainfile], word_vocab; maxlines=1000, ulimit=ulimit)
        ids = nextbatch(ptb, sdict, word_vocab, o[:batchsize]; ulimit=ulimit, maxlines=maxlines)
        while ids != nothing
            train(m, s, ids, opts, tloss)
            ids = nextbatch(ptb, sdict, word_vocab, o[:batchsize]; ulimit=ulimit, maxlines=maxlines)
        end
        tperp = exp(mean(tloss))
        dperp = devperp(m, sdev, dev)
        println("Epoch $i | dperp $dperp | trainperp $tperp")
        flush(STDOUT)
        close(ptb)
    end
end
!isinteractive() && main(ARGS)
