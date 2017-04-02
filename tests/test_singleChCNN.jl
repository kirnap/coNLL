# given a single word of length l and create char level representation
using Knet
include("../util/chproces.jl")
function test_singleChCNN()
    atype = gpu() >= 0 ? KnetArray{Float32} : Array{Float32}
    ch1 = create_chvocab("../ptb/ptb.vocab")

    # minibatched single length words
    fake_word = "util"; fake_word2 = "life"; data = [fake_word, fake_word2, fake_word];

    ######### hyperparemeters ###########
    # l predefined constant word length, d constant embedsize, w window length(need to be less than word length)
    l=4; d=5; w=2; batchsize=length(data); filter_bank = 2;

    # single filter implementation, h has to be in the dimension of (window length by char lookup length), for test purpose
    h = convert(atype, randn(w, d, 1, 1));
    
    # multiple filter implementation, in paper they claim that they have varying w filters
    h2 = convert(atype, randn(w, d, 1, filter_bank))
    # bias implementation
    bias = convert(atype, zeros(1,1, filter_bank, 1))

    
    ###### char-LOOKUP definition #########
    chsize=length(ch1);
    chembed = convert(atype, randn(chsize, d));


    ######## MULTIPLE WORD CNN-BASED Character level representation ########
    all_words = []  # each item of it has the the single word-char based embeddings
    for word in data
        wrep = map(x->ch1[x], collect(word))
        push!(all_words, chembed[wrep, :])
    end
    batchembed = hcat(all_words...)
    brep_conv = reshape(batchembed, (l, d, 1, batchsize))


    ####### CHECKS the multiple batch reshaping operation #####
    (_, _, _, bsize) = size(brep_conv)
    bcheck = convert(Array{Float32}, brep_conv) # since there is no getindex in multiple dim KnetArray
    for i=1:bsize
        @assert(all_words[i] == bcheck[:,:,1,i])
    end
    info("Multiple batch correction passed")

    
    fmap = conv4(h, brep_conv)
    

    ######## CHECK multiple dimension convolution operation ########
    scovs = Any[]
    for item in all_words
        i = reshape(item, (l, d, 1, 1))
        push!(scovs, conv4(h, i))
    end
    # Since there is no indexing in KnetArray, unfortunate testing
    scovs = map(x->convert(Array{Float32}, x), scovs)
    fcheck = convert(Array{Float32}, fmap)
    for i=1:length(scovs)
        @assert(fcheck[:, :, 1, i] == reshape(scovs[i], (l-w+1, 1)))
    end
    info("Each feature representation test passed")
    #####



    # equation 5 in the paper
    x = pool(tanh(conv4(h2, brep_conv) .+ bias))
    x = mat(x)'
    @assert(size(x) == (batchsize, filter_bank))
    info("Batch testing passed each instance in a single row")
    return (brep_conv, h2, h, bias)


    


    
    
    
end
!isinteractive() && test_singleChCNN()

