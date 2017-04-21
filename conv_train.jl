using Knet, JLD, ArgParse
include("util/infst2.jl")
include("models/char_conv.jl")
function main(args=ARGS)
    s = ArgParseSettings()
    s.exc_handler = ArgParse.debug_handler
    @add_arg_table s begin
        ("--trainfile"; help="Infinite stream training file")
        ("--devfile"; help="dev data to test perplexity")
        ("--vocabfile"; help="Vocabulary file to train a model")
        ("--wordsfile"; required=true; help="Words file used to hold all words")
        ("--savefile"; help="To save the julia model")

        ("--loadmodel"; help="Loadfile for model loading")
        ("--loadvocab"; help="JLD file holds the character and word level vocabulary")

        ("--atype"; default=(gpu() >= 0 ? "KnetArray{Float32}" : "Array{Float32}"))
        ("--hiddens"; arg_type=Int; nargs='+'; default=[300]; help="hidden layer configuration")
        ("--filterbank"; arg_type=Int; default=500; help="number of features")
        ("--windowlen"; arg_type=Int; default=5; help="Window length of convolution filters")
        ("--chembed"; arg_type=Int; default=15)
        ("--batchsize"; arg_type=Int; default=25)
        ("--gclip"; arg_type=Float64; default=5.0)
        
        
    end
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    atype = eval(parse(o[:atype]))
    println("Model file: char_conv.jl")
    
    for (k, v) in o
        println("$k => $v")
    end

    # word data initilaziation
    if o[:loadvocab] == nothing
        word_vocab_out = create_vocab(o[:vocabfile])

        # character level initialization
        # TODO decide how to do unique character counting
    else
        vocabs = load(o[:loadvocab])
        word_vocab_out = vocabs["word_vocab"]
        char_vocab = vocabs["char_vocab"]
    end

    # TODO put vocabulary saving lines

    
    ulimit = 28; maxlines = 200;
    word_vocab_all = create_vocab(o[:wordsfile])
    dstream = open(o[:trainfile]); s = Dict{Int64, Array{Any, 1}}();
    readstream!(dstream, s, word_vocab_out, word_vocab_all; maxlines=1000)

    # initialize convolutional level vocabulary
    conv_vocab = create_conv_vocab(word_vocab_all)
    i2w_all_conv = Array(AbstractString, length(conv_vocab));
    for (k, v) in conv_vocab; i2w_all_conv[v] = k; end;


    # model initialization
    chsize = length(char_vocab); wordvsize = length(word_vocab_out)
    lw = longest_word(conv_vocab) # longest word in vocabulary
    pwind = lw + o[:chembed] - 1 # pooling window size

    m = initmodel(atype, o[:hiddens], o[:windowlen], o[:filterbank], o[:chembed], chsize, wordvsize)
    states = initstate(atype, o[:hiddens], o[:batchsize])
    opts = oparams(m, Adam; gclip=o[:gclip])

    loss = this_loss = 0
    ids = nextbatch(dstream, s, word_vocab_out, word_vocab_all, o[:batchsize]; maxlines=maxlines, ulimit=ulimit)
    bcount = 1
    println("Training started..."); flush(STDOUT);
    while ids != nothing
        this_loss = train(m, states, ids, i2w_all_conv, char_vocab, lw, o[:chembed], pwind, opts)

        # running average loss
        if bcount < 100
            loss = (this_loss + (bcount - 1)*loss) / bcount
        else
            loss = loss * 0.99 + this_loss * 0.01
        end

        # checkpoints and save
        if bcount % 4000 == 0
            perp = exp(loss)
            println("Running average perlexity is $perp")
            moc = convertmodel(m)
            save(o[:savefile], "model", moc)
            flush(STDOUT)
        end

        ids = nextbatch(dstream, s, word_vocab_out, word_vocab_all, o[:batchsize]; maxlines=maxlines, ulimit=ulimit)
        bcount += 1
        @show exp(loss)
        (bcount == 10000) && break


    end
    
end
!isinteractive() && main(ARGS)
