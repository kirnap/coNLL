# To test different word lengths
using Knet
include("../util/chproces.jl")

function test_different_words_CNN()
    atype = gpu() >= 0 ? KnetArray{Float32} : Array{Float32}
    ch1 = create_chvocab("../ptb/ptb.vocab")
    
    fake_word = "omer"; fake_word2 = "utilitiy"; fake_word3 = "life"; data = [fake_word, fake_word2, fake_word3];

    # l predefined constant word length, d constant embedsize, w window length(need to be less than word length)
    l=8; d=5; w=2; batchsize=length(data); filter_bank = 2;

    # filter bank, bias initialization
    H = convert(atype, randn(w, d, 1, filter_bank)); bias = convert(atype, zeros(1, 1, filter_bank, 1))


    ###### char-LOOKUP definition #########
    chsize=length(ch1);
    chembed = convert(atype, randn(chsize, d));

    ####### padding with zero test embeddings of each word, 
    each_word_embedding = []
    each_word_embedding_zero_padded = []
    for word in data
        wr = map(x->ch1[x], collect(word))
        x1 = chembed[wr,:]
        x = x1
        if length(wr) < l
            (rows, cols) = l - length(wr), d
            pad = atype(zeros(rows, cols))
            x = vcat(x, pad)
        end
        push!(each_word_embedding, x1)
        push!(each_word_embedding_zero_padded, x)
    end
    @assert(batchsize*l*d == mapreduce(length, +, 0, each_word_embedding_zero_padded))
    info("zero padding test passed")
    batchembed = hcat(each_word_embedding_zero_padded...)
    brep_conv = reshape(batchembed, (l, d, 1, batchsize))


    ####### CHECKS the multiple batch reshaping operation #####
    (_, _, _, bsize) = size(brep_conv)
    bcheck = convert(Array{Float32}, brep_conv) # since there is no getindex in multiple dim KnetArray
    for i=1:bsize
        @assert(each_word_embedding_zero_padded[i] == bcheck[:,:,1,i])
    end
    info("Multiple batch correction passed")

    # equation 5 in the paper
    pool_winsize = l + w - 1
    x = pool(tanh(conv4(H,brep_conv) .+ bias); window=pool_winsize)
    return (x, H, brep_conv)
end
!isinteractive() && test_different_words_CNN()
