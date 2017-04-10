using Knet, ArgParse, JLD
include("util/infst2.jl")
include("models/charfinlsos_model.jl")

function main(args=ARGS)
    s = ArgParseSettings()
    s.exc_handler = ArgParse.debug_handler
    @add_arg_table s begin
        ("--trainfile"; required=true; help="Infinite stream training file")
        ("--devfile"; help="dev data to test perplexity")
        ("--vocabfile"; required=true; help="Vocabulary file to train a model")
        ("--wordsfile"; required=true; help="Words file used to hold all words")
        ("--vosave"; help="Words file used to hold all words")
        ("--loadfile"; help="Loadfile for model loading")
        ("--atype"; default=(gpu() >= 0 ? "KnetArray{Float32}" : "Array{Float32}"))
        ("--hiddens"; arg_type=Int; nargs='+'; default=[300]; help="hidden layer configuration")
        ("--embedding"; arg_type=Int; default=350)
        ("--chembedding"; arg_type=Int; default=15)
        ("--batchsize"; arg_type=Int; default=25)
        ("--gclip"; arg_type=Float64; default=5.0)
        ("--savefile"; help="To save the julia model")
    end
    
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    atype = eval(parse(o[:atype]))
    println("Model file: charfinlsos_model.jl")

    for (k, v) in o
        println("$k => $v")
    end

    # word data initialization
    ulimit = 28; maxlines = 500;
    word_vocab_out = create_vocab(o[:vocabfile])
    word_vocab_all = create_vocab(o[:wordsfile])
    i2w_all = Array(AbstractString, length(word_vocab_all))
    for (k, v) in word_vocab_all; i2w_all[v] = k;end;
    dstream = open(o[:trainfile]); s = Dict{Int64, Array{Any, 1}}();
    readstream!(dstream, s, word_vocab_out, word_vocab_all; maxlines=1000)
    
    

    # character level initialization
    char_vocab = create_chvocab(word_vocab_out)
    info("Character vocabulary length: $(length(char_vocab))")
    if o[:vosave] != nothing
        x = o[:vosave]
        info("saving vocabularies to $x")
        save(o[:vosave], "char_vocab", char_vocab, "word_vocab", word_vocab_out)
    end
    flush(STDOUT)
    
    # model initialization
    charhiddens = [o[:embedding]]; wordvsize = length(word_vocab_out); chvsize = length(char_vocab);
    if o[:loadfile] == nothing
        m = initmodel(atype, o[:hiddens], charhiddens, o[:chembedding], wordvsize, chvsize)
    else # to fix broken trainings
        x = load(o[:loadfile])
        m = revconvert(x["model"]) # for reverse converting the model
    end
    char_states = initstate(atype, charhiddens, o[:batchsize])
    states = initstate(atype, o[:hiddens], o[:batchsize])
    opts = oparams(m, Adam; gclip=o[:gclip])

    loss = this_loss = 0

    ids = nextbatch(dstream, s, word_vocab_out, word_vocab_all, o[:batchsize]; maxlines=100, ulimit=ulimit)
    bcount = 1
    info("Training started...")
    while ids !=nothing
        this_loss = train(m, char_states, states, ids, i2w_all, char_vocab, opts)

        # running average loss
        if bcount < 100
            loss = (this_loss + (bcount - 1)*loss) / bcount
        else
            loss = loss * 0.99 + this_loss * 0.01
        end

        # checkpoints and save
        if bcount % 4000 == 0
            perp = exp(loss)
            println("Running average loss is $perp")
            moc = convertmodel(m)
            save(o[:savefile], "model", moc)
            flush(STDOUT)
        end

        ids = nextbatch(dstream, s, word_vocab_out, word_vocab_all, o[:batchsize]; maxlines=300, ulimit=ulimit)
        bcount += 1
    end

end
!isinteractive() && main(ARGS)
