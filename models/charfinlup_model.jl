# Single layer character based lstm gets lookup inputs and gives its final layer to bilstm
include("../util/prop.jl")


function initmodel(atype, hiddens, charhidden, charembed, wordvocab, charvocab, init=xavier)
    model = Dict{Symbol, Any}()
    wordembedding = charhidden[1]
    model[:forw] = initweights(atype, hiddens, wordembedding, init)
    model[:back] = initweights(atype, hiddens, wordembedding, init)
    model[:char] = initweights(atype, charhidden, charembed, init)
    model[:cembed] = atype(init(charvocab, charembed))
    model[:soft] = [ atype(2hiddens[end], wordvocab), atype(init(1, wordvocab)) ]
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


function charembed(mchar, mcembed, states, wids, i2w, ch, atype)
    schar = copy(states)
    
    (data, masks) = cbatchlookup(wids, i2w, ch)
    h = zeros(similar(schar[1]))
    for (c, m) in zip(data, masks)
        embed = mcembed[c, :]
        mbon = convert(atype, m)
        h = chforw(mchar, schar, embed; mask=mbon)
    end
    return h
end


function charbilstm(model, chstates, states, sequence, i2w, chvocab, lval=[])
    total = 0.0
    count = 0
    atype = typeof(states[1])

    # extract the embeddings on character reading
    embeddings = Array(Any, length(sequence))
    for i=1:length(sequence)
        embeddings[i] = charembed(model[:char], model[:cembed], chstates, sequence[i], i2w, chvocab, atype)
    end

    # forward lstm
    fhiddens = Array(Any, length(sequence))
    sf = copy(states)
    for i=1:length(sequence)-1
        h = forward(model[:forw], sf, embeddings[i])
        fhiddens[i+1] = copy(h)
    end
    fhiddens[1] = zeros(similar(fhiddens[2]))

    
    # backward lstm
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
    val = - total / count
    push!(lval, AutoGrad.getval(val))
    return val
end


gradcharbilstm = grad(charbilstm)


function train(model, chstates, states, sequence, i2w, chvocab, lval, opts)
    gloss = gradcharbilstm(model, chstates, states, sequence, i2w, chvocab, lval)
    update!(model, gloss, opts)
    return lval
end


function devperp(m, schar, sdev, dev, i2w, ch1)
    devloss = []
    for d in dev
        charbilstm(m, schar, sdev, d, i2w, ch1, devloss)
    end
    return exp(mean(devloss))
end
