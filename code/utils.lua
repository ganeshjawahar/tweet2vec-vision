--[[
Utility function used by multiple lua classes.
--]]

require 'torch'

local utils={}

-- Function to check if the input is a valid number
function utils.isNumber(a)
	if tonumber(a) ~= nil then
		return true
	end
	return false
end

-- Function to trim the string
function utils.trim(s)
  return (s:gsub("^%s*(.-)%s*$", "%1"))
end

-- Function to tokenize a tweet
function utils.tokenize(tweet)
	local res={}
	for word in string.gmatch(tweet,'%S+') do
		if string.sub(word,1,1)=='@' and #word>1 then
			table.insert(res,'<USER>')
		elseif #word>4 and string.sub(word,1,4)=='http' then
			table.insert(res,'<URL>')
		elseif utils.isNumber(word)==true then
			table.insert(res,'<NUM>')
		else
			table.insert(res,word)
		end
	end
	return res
end

-- Function to normalize url
function utils.normalizeUrl(tweet)
	local res=''
	for word in string.gmatch(tweet,'%S+') do
		if #word>4 and string.sub(word,1,4)=='http' then
			res=res..'<URL> '
		else
			res=res..word..' '
		end
	end
	return utils.trim(res)
end

-- Function to get all ngrams
function utils.getNgrams(tweet,n,pad)
	local res={}
	local tokens=utils.padTokens(utils.splitByChar(tweet,' '),pad) --assuming the tweets is already tokenized by Gimpel.
	for i=1,(#tokens-n+1) do
		local word=''
		for j=i,(i+(n-1)) do
			word=word..tokens[j]..' '
		end
		word=utils.trim(word)
		table.insert(res,word)
	end
	return res
end

-- Function to pad tokens.
function utils.padTokens(tokens,pad)
	local res={}

	-- Append begin tokens
	for i=1,pad do
		table.insert(res,'<bpad-'..i..'>')
	end

	for _,word in ipairs(tokens) do
		table.insert(res,word)
	end

	-- Append end tokens
	for i=1,pad do
		table.insert(res,'<epad-'..i..'>')
	end

	return res
end

-- Function to split a string by given char.
function utils.splitByChar(str,inSplitPattern)
	utils.trim(str)
	outResults={}
	local theStart = 1
	local theSplitStart,theSplitEnd=string.find(str,inSplitPattern,theStart)
	while theSplitStart do
		table.insert(outResults,string.sub(str,theStart,theSplitStart-1))
		theStart=theSplitEnd+1
		theSplitStart,theSplitEnd=string.find(str,inSplitPattern,theStart)
	end
	table.insert(outResults,string.sub(str,theStart))
	return outResults
end

-- Function to save the embeddings
function utils.saveEmbeddings(type,mat,map,f)
	print('Saving '..type..' embeddings...')
	local start=sys.clock()
	local fptr=io.open(f,'w')
	for i=1,(#mat)[1] do
		local line=map[i]..'\t'
		for j=1,(#mat)[2] do
			line=line..mat[i][j]..'\t'
		end
		line=line..'\n'
		fptr:write(line)
	end
	fptr:close()
	print(string.format("Done in %.2f seconds.",sys.clock()-start))
end

-- Function to save the tweet embeddings
function utils.saveTweetEmbeddings(mat,loc)
	local file=io.open(loc,'w')
	for i=1,(#mat)[1] do
		line=''
		for j=1,(#mat)[2] do
			line=line..mat[i][j]..'\t'
		end
		line=utils.trim(line)..'\n'
		file:write(line)
	end
	file:close()
end

-- Function to build vocabulary from the corpus
function utils.buildVocab(config)
	print('Building vocabulary...')
	local start=sys.clock()
	local pad=0
	if config.pad_tweet==1 then pad=(config.wwin/2) end
	
	-- Fill the vocabulary frequency map
	local n=0
	config.total_count=0
	for line in io.lines(config.train_file) do
		local content=utils.splitByChar(line,'\t')
		if #content>=4 then
			local tweet_text=content[4]
			for _,word in ipairs(utils.getNgrams(tweet_text,1,pad)) do
				config.total_count=config.total_count+1

				if config.to_lower==1 then
					word=word:lower()
				end

				-- Fill word vocab.
				if config.vocab[word]==nil then
					config.vocab[word]=1
				else
					config.vocab[word]=config.vocab[word]+1
				end
			end
			n=n+1
		end
	end
	config.corpus_size=n

	-- Discard the words that doesn't meet minimum frequency and create indices.
	for word,count in pairs(config.vocab) do
		if count<config.min_freq then
			config.vocab[word]=nil
		else
			config.index2word[#config.index2word+1]=word
			config.word2index[word]=#config.index2word
		end
	end

	-- Add unknown word
	config.vocab['<UK>']=1
	config.index2word[#config.index2word+1]='<UK>'
	config.word2index['<UK>']=#config.index2word
	config.vocab_size= #config.index2word

	print(string.format("%d words, %d tweets processed in %.2f seconds.",config.total_count,n,sys.clock()-start))
	print(string.format("Vocab size after eliminating words occuring less than %d times: %d",config.min_freq,config.vocab_size))
end

-- Function to create a lookup table
function utils.getLookupTable(row,col,a,b)
	local table=nn.LookupTable(row,col)
	table.weight:uniform(a,b)
	return table
end

-- Function to get word windows from a tweet (with labels)
function utils.getWordWindows(tweet,pad,wwin,word2index,isCenter)
	local windows={}
	local tokens=utils.getNgrams(tweet,1,pad)
	for i=1,((#tokens)-wwin+1) do
		local window=torch.Tensor(wwin-1)
		local target=torch.Tensor(1)
		index=1
		for j=1,wwin do
			local wordId=word2index[tokens[j+(i-1)]]
			if wordId==nil then wordId=word2index['<UK>'] end
			if isCenter==1 and j==(math.ceil(wwin/2)) then
				target[1]=wordId
			elseif isCenter==0 and j==wwin then
				target[1]=wordId
			else
				window[index]=wordId
				index=index+1
			end
		end
		table.insert(windows,{window,target})
	end
	return windows
end

-- Function to sample negative contexts
function utils.sample_contexts(config,context)
	local contexts=torch.Tensor(1+config.neg_samples)
	contexts[1]=context
	local i=0
	while i<config.neg_samples do
		neg_context=config.table[torch.random(config.table_size)]
		if context~=neg_context then
			contexts[i+2]=neg_context
			i=i+1
		end
	end
	return contexts
end

-- Function to get word windows from a tweet (with negative sample)
function utils.getWordWindowsWithNegSamples(tweet,config)
	local windows={}
	local tokens=utils.getNgrams(tweet,1,0)
	for i=1,#tokens do
		local wid=config.word2index[tokens[i]]
		if wid==nil then wid=config.word2index['<UK>'] end
		table.insert(windows,utils.sample_contexts(config,wid))
	end
	return windows
end

-- Function to get word windows from a tweet (without negative sample)
function utils.getWordWindowsWithoutNegSamples(tweet,wwin,word2index)
	local windows={}
	local tokens=utils.getNgrams(tweet,1,0)
	for i=1,((#tokens)-wwin+1) do
		local window=torch.Tensor(wwin)
		index=1
		for j=1,wwin do
			local wordId=word2index[tokens[j+(i-1)]]
			if wordId==nil then wordId=word2index['<UK>'] end
			window[index]=wordId
			index=index+1
		end
		table.insert(windows,window)
	end
	return windows
end

-- Function to convert tensor to string
function utils.convertToString(tensor)
	local row=''
	for j=1,(#tensor)[1] do
		row=row..tensor[j]..'\t'
	end
	row=utils.trim(row)
	return row
end

-- Function to convert string to tensor
function utils.convertTo1DTensor(str)
	local content=utils.splitByChar(str,'\t')
	local tensor=torch.Tensor(#content)
	for i=1,#content do
		tensor[i]=tonumber(content[i])
	end
	return tensor
end

-- Function to load train set into memory
function utils.loadDataToMemory(config,file)
	config.user_map={}
	config.key_list={}
	config.index2tweet={}
	for line in io.lines(file) do
		local content=utils.splitByChar(line,'\t')
		if #content>=4 then
			local key=content[1]
			local tweet=content[4]
			if config.user_map[key]==nil then
				config.user_map[key]={}
				config.key_list[#config.key_list+1]=key
			end
			config.index2tweet[#config.index2tweet+1]=tweet
			table.insert(config.user_map[key],#config.index2tweet)
		end	
	end
end

-- Function to load train set into memory
function utils.loadTensorsToMemory(config,file)
	utils.loadDataToMemory(config,file)
	local pad=0
	if config.pad_tweet==1 then pad=((config.wwin-1)/2) end

	local w_i_1={}
	local w_i_2={}
	local w_o={}
	local total=#config.key_list
	local indices=torch.randperm(total)
	for index=1,total do
		local key=config.key_list[indices[index]]
		local data=config.user_map[key]
		for tweet_index=1,#data do
			local t_id=data[tweet_index]
			local tweet_text=config.index2tweet[t_id]
			local windows=utils.getWordWindows(tweet_text,pad,config.wwin,config.word2index,config.is_center_target)
			for wi,window in ipairs(windows) do
				table.insert(w_i_1,window[1])
				table.insert(w_i_2,t_id)
				table.insert(w_o,window[2])
			end
		end
	end
	
	local t_i={}
	local t_o={}
	indices=torch.randperm(total)
	for index=1,total do
		local key=config.key_list[indices[index]]
		local data=config.user_map[key]
		local windows=utils.getTweetWindows(data,config.twin,config.corpus_size,1)
		for wi,window in ipairs(windows) do
			table.insert(t_i,window[1])
			table.insert(t_o,window[2])
		end
	end

	-- Create the tensors	
	config.word_model_word_context=torch.IntTensor(#w_i_1,config.wwin-1)
	config.word_model_tweet_target=torch.IntTensor(#w_i_2,1)
	config.word_model_word_target=torch.IntTensor(#w_o,1)
	config.tweet_model_context=torch.IntTensor(#t_i,config.twin-1)
	config.tweet_model_target=torch.IntTensor(#t_o,1)
	for i=1,#w_i_1 do
		config.word_model_word_context[i]=w_i_1[i]
		config.word_model_tweet_target[i]=w_i_2[i]
		config.word_model_word_target[i]=w_o[i]
	end
	for j=1,#t_i do
		config.tweet_model_context[j]=t_i[j]
		config.tweet_model_target[j]=t_o[j]
	end

	-- Clean the memory
	config.wm_size=#w_i_1
	config.tm_size=#t_i
	w_i_1=nil
	w_i_2=nil
	w_o=nil
	t_i=nil
	t_o=nil

end

-- Function to get temporal context of a tweet
function utils.getTweetWindows(data,twin,corpus_size,isCenter)
	local windows={}
	local tweets=utils.padTemporalContext(data,twin,corpus_size)
	for i=1,((#tweets)-twin+1) do
		local window=torch.IntTensor(twin-1)
		local target=torch.IntTensor(1)
		index=1
		for j=1,twin do
			local tweetId=tweets[j+(i-1)]
			if isCenter==1 and j==(math.ceil(twin/2)) then
				target[1]=tweetId
			elseif isCenter==0 and j==twin then
				target[1]=tweetId
			else
				window[index]=tweetId
				index=index+1
			end
		end
		table.insert(windows,{window,target})
	end
	return windows
end

-- Function to pad temporal context
function utils.padTemporalContext(tweets,twin,corpus_size)
	local res={}	
	local pad=(twin-1)/2

	-- Append begin tokens
	for i=1,pad do
		table.insert(res,(corpus_size-twin+i+1))
	end

	for _,tweet in ipairs(tweets) do
		table.insert(res,tweet)
	end

	-- Append end tokens
	for i=pad+1,2*pad do
		table.insert(res,(corpus_size-twin+i+1))
	end

	return res
end

-- Function to initalize word weights
function utils.initWordWeights(config,file)
	print('initializing the pre-trained embeddings...')
	local start=sys.clock()
	local ic=0
	for line in io.lines(file) do
		local content=utils.splitByChar(line,' ')
		local word=content[1]
		if config.word2index[word]~=nil then
			local tensor=torch.Tensor(#content-1)
			for i=2,#content do
				tensor[i-1]=tonumber(content[i])
			end
			config.word_vecs.weight[config.word2index[word]]=tensor
			ic=ic+1
		end
	end
	print(string.format("%d out of %d words initialized.",ic,#config.index2word))
	print(string.format("Done in %.2f seconds.",sys.clock()-start))
end

-- Function to build frequency-based tree for Hierarchical Softmax
function utils.create_frequency_tree(freq_map)
	binSize=100
	local ft=torch.IntTensor(freq_map)
	local vals,indices=ft:sort()
	local tree={}
	local id=indices:size(1)
	function recursiveTree(indices)
		if indices:size(1)<binSize then
			id=id+1
			tree[id]=indices
			return
		end
		local parents={}
		for start=1,indices:size(1),binSize do
			local stop=math.min(indices:size(1),start+binSize-1)
			local bin=indices:narrow(1,start,stop-start+1)
			assert(bin:size(1)<=binSize)
			id=id+1
			table.insert(parents,id)
			tree[id]=bin
		end
		recursiveTree(indices.new(parents))
	end
	recursiveTree(indices)
	return tree,id
end

-- Function to create word map (for Softmaxtree)
function utils.create_word_map(vocab,index2word)
	word_map={}
	for i=1,#index2word do
		word_map[i]=vocab[index2word[i]]
	end
	return word_map
end

-- Function to create tweet map (for Softmaxtree)
function utils.create_tweet_map(tweet_count)
	tweet_map={}
	for i=1,tweet_count do
		tweet_map[i]=1
	end
	return tweet_map
end

return utils