# Word embeddings are extracted based on character convolutional network
include("../util/prop.jl")

function initmodel(atype, hiddens, windowlen, filterbank, chembedding, charvocab, outvocab, init=xavier)
    model = Dict{Symbol, Any}()
    model[:cembed] = atype(init(charvocab, chembedding))
    model[:conv] = [ atype(init(windowlen, chembedding, 1, filterbank)),
                     atype(zeros(1, 1, filterbank, 1)) ]
    model[:forw] = initweights(atype, hiddens, filterbank, init)
    model[:back] = initweights(atype, hiddens, filterbank, init)
    model[:soft] = [ atype(init(2hiddens[end], outvocab)), atype(zeros(1, outvocab)) ]
    return model
end

# ugly but works to obtain generic array constructors, in loss function
agtype{T, N}(x::KnetArray{T, N}) = KnetArray{T}
agtype{T, N}(x::Array{T, N}) = Array{T}


"""
mconv[1],[2]: filter bank and bias respectively.
lw : longest word in the vocabulary
d  : character embedding size
wind : window length of the filters
pwin : pooling window size -> lw + d - 1 
"""
function convembed(mconv, mcembed, wids, i2w_all_conv, ch, lw, d, pwin, atype)
    (data, mask) = cbatch4conv(wids, i2w_all_conv, ch, atype)
    batchsize = Int(length(data) / lw)

    to_embed = []
    for i=1:lw:length(data)
        win = data[i:(i+lw-1)]
        x = mcembed[win, :]
        x1 = x .* mask[i:(i+lw-1)]
        push!(to_embed, x1)
    end
    emw = hcat(to_embed...)
    c_k = reshape(emw, (lw, d, 1, batchsize))
    y_k = pool(tanh(conv4(mconv[1], c_k) .+ mconv[2]); window=pwin)
    embedding = mat(y_k)'
    return embedding
end



function convbilstm(model, states, sequence, i2w_all_conv, ch, lw, d, pwin, lval=[])
    total = 0.0
    count = 0
    atype = agtype(states[1])

    # extract embeddings from convolutional layer
    embeddings = Array(Any, length(sequence))
    for i=1:length(sequence)
        embeddings[i] = convembed(model[:conv], model[:cembed], sequence[i], i2w_all_conv, ch, lw, d, pwin, atype)
    end

    # forward lstm
    fhiddens = Array(Any, length(sequence)-2)
    sf = copy(states)
    for i=1:length(sequence)-2
        x = embeddings[i]
        h = forward(model[:forw], sf, x)
        fhiddens[i] = copy(h)
    end

    # backward lstm
    bhiddens = Array(Any, length(sequence)-2)
    sb = copy(states)
    for i=length(sequence):-1:3
        x = embeddings[i]
        h = forward(model[:back], sb, x)
        bhiddens[i-2] = copy(h)
    end

    # concatenate layer
    for i=1:length(fhiddens)
        ypred = hcat(fhiddens[i], bhiddens[i]) * model[:soft][1] .+ model[:soft][2]
        ygold = map(x->x[1], sequence[i+1])
        total += logprob(ygold, ypred)
        count += length(ygold)
    end
    val = - total / count
    push!(lval, AutoGrad.getval(val))
    return val
    
end

gradconvbilstm = grad(convbilstm)


function train(model, states, sequence, i2w_all_conv, ch, lw, d, pwin, opts)
    lval = []
    gloss = gradconvbilstm(model, states, sequence, i2w_all_conv, ch, lw, d, pwin, lval)
    update!(model, gloss, opts)
    return lval[1]
end
