const PAD = '⋮'
const SOW = '↥'
const EOW = 'Ϟ'


function create_chvocab(f::AbstractString)
    res = Dict{Char, Int}(PAD=>1, SOW=>2, EOW=>3)
    stream = open(f)
    for line in eachline(f)
        for char in line
            get!(res, char, 1+length(res))
        end
    end
    (' ' in keys(res)) && delete!(res, ' ')
    return res
end


longest_word{T}(word_vocab::Dict{T, Int}) = findmax(map(length, keys(word_vocab)))[1]


function w2cs(windex::Int32, i2w::Array{AbstractString,1}, chvoc::Dict{Char, Int})
    word = i2w[windex]
    res = Array(Int, length(word))
    for i=1:length(word)
        res[i] = chvoc[word[i]]
    end
    return res
end


function cbatchlookup(wids::Array{Int32, 1}, i2w::Array{AbstractString, 1}, ch::Dict{Char, Int})
    words = map(x->i2w[x], wids)
    critic = findmax(map(length, words))[1]
    batchsize = length(words)
    
    data = Array(Any, critic+2) 
    data[1] = fill!(zeros(Int32, batchsize), ch[SOW])
    masks = Array(Any, critic+2)
    masks[1] = fill!(zeros(Int32, batchsize, 1), Float32(1.0))
    for cursor=1:critic+1 # to pad EOW to the end of the word
        d = Array(Int32, batchsize)
        mask = ones(Float32, batchsize, 1)
        @inbounds for i=1:batchsize
            word = words[i]
            if length(word) < critic
                if length(word) >= cursor
                    d[i] = ch[word[cursor]]
                elseif length(word)+1 == cursor
                    d[i] = ch[EOW]
                else
                    d[i] = ch[PAD]
                    mask[i] = 0
                end
            else
                if cursor>critic
                    d[i] = ch[EOW]
                else
                    d[i] = ch[word[cursor]]
                end
            end
        end
        data[cursor + 1] = d
        masks[cursor + 1] = mask
    end
    return data, masks
end


function cbatch4conv(wids::Array{Int32, 1}, i2w::Array{AbstractString, 1}, ch::Dict{Char, Int})
    words = map(x->i2w[x], wids)
    batchsize = length(words)
    d = Array(Any, batchsize)
    for w=1:batchsize
        word = words[w]
        wlen = length(word)
        inds = Array(Int, wlen+2)
        inds[1] = ch[SOW]
        @inbounds for i=1:wlen
            inds[i+1] = ch[word[i]]
        end
        inds[end] = ch[EOW]
        d[w] = inds
    end
    return d
end


function charbatch(wids::Array{Int32, 1}, i2w::Array{AbstractString, 1}, ch::Dict{Char, Int})
    words = map(x->i2w[x], wids)
    critic = findmax(map(length, words))[1]
    batchsize = length(words)

    data = Array(Any, critic)
    masks = Array(Any, critic)
    for cursor=1:critic
        d = falses(batchsize, length(ch)) #d = Array(Int32, batchsize) no need indexing for charvocab
        mask = zeros(Float32, batchsize, 1)
        for i=1:length(words)
            if length(words[i]) >= cursor
                index = ch[words[i][cursor]]
                d[i, index] = 1
                mask[i] = 1
            else
                index = ch[PAD]
                d[i, index] = 1
            end
        end
        data[cursor] = d
        masks[cursor] = mask
    end
    return (data,masks)
end


function ibuild_word(i2c::Array{Char, 1}, data::Array{Any, 1}, kth::Int; verbose=false)
    word = Any[]
    for i=1:length(data)
        z = find(x->x==true, data[i][kth, :])
        append!(word, i2c[z])
    end
    if verbose
        for item in word; print("$item");end;
        println()
    else
        return word
    end
end


function ibuild_word_lookup(i2c::Array{Char, 1}, data::Array{Any, 1}, kth::Int; verbose=false)
    word = Any[]
    for i=1:length(data)
        z = data[i][kth]
        push!(word, i2c[z])
    end
    if verbose
        for item in word;print("$item");end;println();
    else
        return word
    end
end
