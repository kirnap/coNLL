# Single layer character based lstm, and bilstm for word substitution

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
    model[:forw] = initweights(atype, hiddens, charhidden, init)
    model[:back] = initweights(atype, hiddens, charhidden, init)
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


function charbilstm(model, chstates, states, sequence, i2w, ch1; pdrop=(0,0))
    total = 0.0
    count = 0
    atype = typeof(AutoGrad.getval(model[:soft][1]))

    # get word embeddings
    i2bilstm = []
    for token in sequence
        schar = copy(chstates)
        (data, masks) = charbatch(token, i2w, ch1)
        o1 = []
        i = 1
        for (ch, m) in zip(data, masks)
            chgpu = convert(atype, ch)
            mgpu = convert(atype, m)
            h = chforw(model[:char], schar, chgpu; mask=mgpu)
            push!(o1, h)	
        end
        batch_words = vcat(o1...)
        push!(i2bilstm, batch_words)
    end

    return i2bilstm
end
