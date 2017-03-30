using Knet
include("../model.jl")
include("../util/infst.jl")
include("../util/chproces.jl")
function feed_model()
    atype = (gpu() >= 0 ? KnetArray{Float32} : Array{Float32})

    # word model initialization
    ptb = open("../ptb/ptb.train.txt")
    sdict = Dict{Int64, Array{Any, 1}}(); ulimit=40; maxlines = 500; batchsize = 6;
    word_vocab = create_vocab("../ptb/ptb.vocab")
    readstream!(ptb, sdict, word_vocab;maxlines=1000, ulimit=ulimit)
    ids = nextbatch(ptb, sdict, word_vocab, batchsize; ulimit=ulimit, maxlis=maxlines)
    i2w = Array(AbstractString, length(word_vocab));
    for (k, v) in word_vocab; i2w[v] = k;end;


    # character settings
    ch1 = create_chvocab("../ptb/ptb.vocab")
    i2c = Array(Char, length(ch1));
    for (k, v) in ch1; i2c[v]=k ;end;
    wids = ids[rand(1:length(ids))]
    (data, masks) = charbatch(wids, i2w, ch1) # data contains multiple time step input

    
    # for sanity don't care with that part after first trial
    words = map(x->i2w[x], wids); for word in words; print("$word\n");end;println("---");
    pchs = Any[]
    for i=1:length(data)
        item = data[i];mitem = masks[i]
        chars = Any[]
        rows, cols = size(item)
        for i=1:rows;z = find(x->x==true, item[i, :]); append!(chars, i2c[z]);end;
        ds = Any[];
        for k=1:length(chars)
            x = (chars[k], mitem[k])
            push!(ds, x)
        end
        push!(pchs, ds)
        #k=1;for character in chars;println("$character $(mitem[k])");k+=1;end;
    end
    for i=1:length(pchs[1]); for item in pchs; print("$(item[i]) ");end;println();end;
    #==# 

    # model parameters initialization
    hiddens = [256]; charhidden = [256]; charvocab = length(ch1); wordvocab = length(word_vocab);


    m = initmodel(atype, hiddens, charhidden, charvocab, wordvocab)
    schar = initstate(atype, charhidden, batchsize)
    
    # crucial step to make model feasible

    # global omer_i2c = i2c open that line for testing
    for (item,mask) in zip(data, masks)
        cbon = convert(atype, item)
        chmask = convert(atype, mask)
        masbon = convert(atype, chmask)
        #lval = chlstm(m[:char][1], m[:char][2], schar[1], schar[2], cbon; mask=masbon)[1]
        lval = chforw(m[:char], schar, cbon; mask=masbon)
        @show sum(lval)
    end
    
end
!isinteractive() && feed_model()

#add these lines to model.jl file in lines just after chlstm definition
#input_check = convert(Array{Float32}, input);mask_check = convert(Array{Float32}, mask)
#chars=Any[];for i=1:size(input)[1];z = find(x->x==true, input_check[i, :]); append!(chars, omer_i2c[z]);end;
# println("Input to char lstm! ")
#for k=1:length(chars);println("$(chars[k]) $(mask_check[k])");end;
