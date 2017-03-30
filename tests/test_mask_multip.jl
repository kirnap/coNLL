using Knet
include("../util/chproces.jl")
include("../util/infst.jl")
# test mask multiplication
function initstate(atype, hiddens, batchsize)
    state = Array(Any, 2length(hiddens))
    for k=1:length(hiddens)
        state[2k-1] = atype(zeros(batchsize, hiddens[k]))
        state[2k] = atype(zeros(batchsize, hiddens[k]))
    end
    return state
end


function initweights(atype, hiddens, embedding, vocab, init=xavier)
    weights = Array(Any, 2length(hiddens))
    input = embedding
    for k = 1:length(hiddens)
        weights[2k-1] = init(input+hiddens[k], 4hiddens[k])
        weights[2k] = zeros(1, 4hiddens[k])
        weights[2k][1:hiddens[k]] = 1 # forget gate bias
        input = hiddens[k]
    end
    return map(w->convert(atype, w), weights)
end


function lstm(weight, bias, hidden, cell, input; mask=nothing)
    gates   = hcat(input,hidden) * weight .+ bias
    if mask != nothing
        gates = gates .* mask
    end
    hsize   = size(hidden,2)
    forget  = sigm(gates[:,1:hsize])
    ingate  = sigm(gates[:,1+hsize:2hsize])
    outgate = sigm(gates[:,1+2hsize:3hsize])
    change  = tanh(gates[:,1+3hsize:end])
    cell    = cell .* forget + ingate .* change
    hidden  = outgate .* tanh(cell)
    return (hidden,cell)
end

function fake_loss(p, s, i; mask=nothing)
    x = lstm(p[1], p[2], s[1], s[2], i; mask=mask)
    return sum(x[1])
end

l_fake = grad(fake_loss)


function test_maskmul()
    atype = Array{Float32}
    ptb = open("../ptb/ptb.train.txt")
    sdict = Dict{Int64, Array{Any, 1}}();
    ulimit = 40
    maxlines = 500
    batchsize = 6
    word_vocab = create_vocab("../ptb/ptb.vocab")
    readstream!(ptb, sdict, word_vocab;maxlines=1000, ulimit=ulimit)
    ids = nextbatch(ptb, sdict, word_vocab, batchsize; ulimit=ulimit, maxlis=maxlines)

    i2w = Array(AbstractString, length(word_vocab))
    for (k, v) in word_vocab; i2w[v] = k;end;

    # character settings
    ch1 = create_chvocab("../ptb/ptb.vocab")
    i2c = Array(Char, length(ch1));
    for (k, v) in ch1; i2c[v]=k ;end;
    wids = ids[rand(1:length(ids))]
    (data, masks) = charbatch(wids, i2w, ch1)
    seq = data[end]
    
    # initialize lstm
    param = initweights(atype, [15], length(ch1), length(ch1))
    s = initstate(atype, [15], batchsize)
    fake_mask = zeros(similar(masks[end]))
    (h , c) = lstm(param[1], param[2], s[1], s[2], seq; mask =masks[end])
    gs = l_fake(param, s, seq; mask=fake_mask)
    @show gs # here you expect to see zero gradient if the mask is fake_mask
    return (h, c, gs)
end
!isinteractive() && test_maskmul()
