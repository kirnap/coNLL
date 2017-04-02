# given a single word of length l and create char level representation
using Knet
include("../util/chproces.jl")
function test_singleChCNN()
    atype = gpu() >= 0 ? KnetArray{Float32} : Array{Float32}
    ch1 = create_chvocab("../ptb/ptb.vocab")
    fake_word = "util"; wrep = map(x->ch1[x], collect(fake_word))

    
    # character embedding matrix
    ces = 5; chsize=length(ch1); chembed = convert(atype, randn(chsize, ces));
    crep = chembed[wrep, :]
    rows, cols = size(crep)
    crep_conv = reshape(crep ,(rows, cols, 1, 1))

    # h has to be in the dimension of (window length by all chars)
    h = convert(atype, randn(2,ces,1,1))

    # desired feature map
    fmap = conv4(h, crep_conv)
    
    return (fmap, crep_conv, h)

    
    
    
end
!isinteractive() && test_singleChCNN()

