# Single layer character based lstm gets the final hidden state of the charlstm, and bilstm for word substitution
include("../util/prop.jl")

function initmodel(atype, hiddens, charhidden, charvocab, wordvocab, init=xavier)
    model = Dict{Symbol, Any}()
    wordembedding = charhidden[1] # the output of the charlstm is the input embedding of the bilstm
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

gradcharbilstm = grad(charbilstm)
