# Single layer character based lstm gets the final hidden state of the charlstm, and bilstm for word substitution

# lstm weights initialization
# w[2k-1], w[2k] : weight and bias for kth layer respectively
function initweights(atype, hiddens, embedding, init=xavier)
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


# state initialization
# s[2k-1], s[2k] : hidden and cell respectively
function initstate(atype, hiddens, batchsize)
    state = Array(Any, 2length(hiddens))
    for k=1:length(hiddens)
        state[2k-1] = atype(zeros(batchsize, hiddens[k]))
        state[2k] = atype(zeros(batchsize, hiddens[k]))
    end
    return state
end


function initmodel(atype, hiddens, charhidden, charvocab, wordvocab, init=xavier)
    model = Dict{Symbol, Any}()
    wordembedding = charhidden[1]
    model[:forw] = initweights(atype, hiddens, wordembedding, init)
    model[:back] = initweights(atype, hiddens, wordembedding, init)
    model[:char] = initweights(atype, charhidden, charvocab, init)
    model[:soft] = [ atype(init(2hiddens[end], wordvocab)), atype(init(1, wordvocab)) ]
    return model
end


function chlstm(weight, bias, hidden, cell, input; mask=nothing)
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


function chforw(weight, states, input; mask=nothing)
    x = input
    for i=1:2:length(states)
        (states[i], states[i+1]) = chlstm(weight[i], weight[i+1], states[i], states[i+1], x; mask=mask)
        x = states[i]
    end
    return x
end


function lstm(weight,bias,hidden,cell,input)
    gates   = hcat(input,hidden) * weight .+ bias
    hsize   = size(hidden,2)
    forget  = sigm(gates[:,1:hsize])
    ingate  = sigm(gates[:,1+hsize:2hsize])
    outgate = sigm(gates[:,1+2hsize:3hsize])
    change  = tanh(gates[:,1+3hsize:end])
    cell    = cell .* forget + ingate .* change
    hidden  = outgate .* tanh(cell)
    return (hidden,cell)
end


# multilayer lstm forward, returns the final hidden
function forward(weight, states, input)
    x = input
    for i=1:2:length(states)
        (states[i], states[i+1]) = lstm(weight[i], weight[i+1], states[i], states[i+1], x)
        x = states[i]
    end
    return x
end


function charembed(mchar, states, words, i2w, ch, atype)
    schar = copy(states)
    
    (data, masks) = charbatch(words, i2w, ch) # create minibatches of characters
    h = zeros(similar(schar[1]))
    for (c, m) in zip(data, masks)
        cbon = convert(atype, c)
        mbon = convert(atype, m)
        h = chforw(mchar, schar, cbon; mask=mbon)
    end
    return h
end


function logprob(output, ypred)
    nrows,ncols = size(ypred)
    index = similar(output)
    @inbounds for i=1:length(output)
        index[i] = i + (output[i]-1)*nrows
    end
    o1 = logp(ypred,2)
    o2 = o1[index]
    o3 = sum(o2)
    return o3
end


function charbilstm(model, chstates, states, sequence, i2w, chvocab)
    total = 0.0
    count = 0
    atype = typeof(states[1])

    # extract embeddings based on character reading
    embeddings = Array(Any, length(sequence))
    for i=1:length(sequence)
        embeddings[i] = charembed(model[:char], chstates, sequence[i], i2w, chvocab, atype)
    end

    # forward lstm
    fhiddens = Array(Any, length(sequence))
    sf = copy(states)
    for i=1:length(sequence)-1
        h = forward(model[:forw], sf, embeddings[i])
        fhiddens[i+1] = copy(h)
    end
    fhiddens[1] = zeros(similar(fhiddens[2]))

    # bacward lstm
    bhiddens = Array(Any, length(sequence))
    sb = copy(states)
    for i=length(sequence):-1:2
        h = forward(model[:back], sb, embeddings[i])
        bhiddens[i-1] = copy(h)
    end
    bhiddens[end] = zeros(similar(bhiddens[2]))

    # merge layer
    for i=1:length(fhiddens)
        ypred = hcat(fhiddens[i], bhiddens[i]) * model[:soft][1] .+ model[:soft][2]
        total += logprob(sequence[i], ypred)
        count += length(sequence[i])
    end
    return - total / count

end
