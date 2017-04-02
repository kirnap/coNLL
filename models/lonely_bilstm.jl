# Word embeddings are extracted from embedding matrices
include("../util/prop.jl")

function initmodel(atype, hiddens, embedding, vocabsize, init=xavier)
    model = Dict{Symbol, Any}()
    model[:forw] = initweights(atype, hiddens, embedding, init)
    model[:back] = initweights(atype, hiddens, embedding, init)
    model[:fembed] = atype(init(vocabsize, embedding))
    model[:bembed] = atype(init(vocabsize, embedding))
    model[:soft] = [ atype(init(2hiddens[end], vocabsize)), atype(init(1, vocabsize)) ]
    return model
end


function bilstm(model, states, sequence, lval=[])
    total = 0.0
    count = 0

    # forward lstm
    fhiddens = Array(Any, length(sequence))
    sf = copy(states)
    for i=1:length(sequence)-1
        x = model[:fembed][sequence[i], :]
        h = forward(model[:forw], sf, x)
        fhiddens[i+1] = copy(h)
    end
    fhiddens[1] = zeros(similar(fhiddens[2]))

    # backward lstm
    bhiddens = Array(Any, length(sequence))
    sb = copy(states)
    for i=length(sequence):-1:2
        x = model[:bembed][sequence[i], :]
        h  = forward(model[:back], sb, x)
        bhiddens[i-1] = copy(h)
    end
    bhiddens[end] = zeros(similar(bhiddens[2]))

    # concatenate layer
    for i=1:length(fhiddens)
        ypred = hcat(fhiddens[i], bhiddens[i]) * model[:soft][1] .+ model[:soft][2]
        total += logprob(sequence[i], ypred)
        count += length(sequence[i])
    end
    val = - total / count
    push!(lval, AutoGrad.getval(val))
    return val
end

gradbilstm = grad(bilstm)


function train(model, state, sequence, opts, lval)
    gloss = gradbilstm(model, state, sequence, lval)
    update!(model, gloss, opts)
end


function devperp(model, state, dev)
    devloss = []
    for d in dev
        bilstm(model, state, d, devloss)
    end
    return exp(mean(devloss))
end


