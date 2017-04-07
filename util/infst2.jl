# Infinite stream processing version 2, don't give <unk> as an input to model
SOS = "<s>"
EOS = "</s>"
UNK = "<unk>"


function create_vocab(vocabfile::AbstractString)
    ercount = 0
    result = Dict{AbstractString, Int}(SOS=>1, EOS=>2, UNK=>3)
    open(vocabfile) do f
        for line in eachline(f)
            words= split(line)
            try
                @assert(length(words) == 1, "The vocabulary file seems broken")
            catch e
                ercount += 1
                if (length(words) == 0)
                    if (ercount == 5)
                        info("There is empty line in vocabulary file")
                        ercount = 0
                    end
                    continue
                else
                    warn("Something unexpected happen in vocabfile")
                end
            end
            word = words[1]
            get!(result, word, 1+length(result))
        end
    end
    return result
end



"""
Reads from infinite stream modifies the s paramater in the light of sequence lengths,
it modifies the s argument and such that it contains what i called information tuple:
(outid, inid) -> outid is the the output vocabulary to use in softmax, inid is the input vocabulary to use in char level lstm

"""
function readstream!(f::IO,
                     s::Dict{Int64, Array{Any, 1}},
                     vocab::Dict{AbstractString, Int64},
                     realvocab::Dict{AbstractString, Int64};
                     maxlines=100, llimit=3, ulimit=300)
    
    k = 0
    while (k != maxlines)
        eof(f) && return false
        words = split(readline(f))
        wlen = length(words)
        (wlen < llimit) || (wlen > ulimit-2) && continue
        seq = Array(Tuple{Int32, Int32}, wlen+2)
        seq[1] = (vocab[SOS], vocab[SOS])
        for i=1:wlen
            word = words[i]
            index = get(vocab, word, vocab[UNK])
            realindex = get(realvocab, word, realvocab[UNK]) # no unking activity desired
            seq[i+1] = (index, realindex)
        end
        seq[end] = (vocab[EOS], vocab[EOS])
        skey = wlen+2
        (!haskey(s, skey)) && (s[skey] = Any[])
        push!(s[skey], seq)
        k += 1
    end
end


function mbatch(sequences::Array{Any, 1}, batchsize::Int)
    seqlen = length(sequences[1])
    data = Array(Any, seqlen)
    for cursor=1:seqlen
        d = Array(Tuple{Int32, Int32}, batchsize)
        for i=1:batchsize
            d[i] = sequences[i][cursor]
        end
        data[cursor] = d
    end
    return data
end


function nextbatch(f::IO,
                   sdict::Dict{Int64, Array{Any, 1}},
                   vocab::Dict{AbstractString, Int64},
                   realvocab::Dict{AbstractString, Int64},
                   batchsize;
                   o...)
    
    slens = collect(filter(x->length(sdict[x])>=batchsize, keys(sdict)))
    if length(slens) < 10
        readstream!(f, sdict, vocab, realvocab; o...)
        slens = collect(filter(x->length(sdict[x])>=batchsize, keys(sdict)))
        if length(slens) == 0
            return nothing
        end
    end
    slen = rand(slens)
    sequence = sdict[slen][1:batchsize]
    deleteat!(sdict[slen], 1:batchsize)
    return mbatch(sequence, batchsize)
end


"""
bytpe: out -> give "unk"ed version of the word, give normal version otherwise.
"""
function ibuild_sentence(i2w::Array{AbstractString, 1}, sequence::Array{Any, 1}, kth::Int; btype=:out, verbose=true)
    sentence = Any[]
    for i=1:length(sequence)
        z = sequence[i][kth]
        index = ((btype == :out) ? z[1] : z[2])
        push!(sentence, i2w[index])
    end
    if verbose
        for item in sentence; print("$item ");end;
    else
        return sentence
    end

end
