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
	str=utils.trim(str)
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
function utils.loadDataToMemory(config)
	config.user_map={}
	config.key_list={}
	config.index2tweettext={}
	config.index2tweetid={}
	config.tweet2index={}	
	config.word_model_size=0
	config.word_model_pad=0; if config.pad_tweet==1 then config.word_model_pad=((config.wwin-1)/2) end
	for line in io.lines(config.train_file) do
		local content=utils.splitByChar(line,'\t')
		if #content>=4 then
			local key=content[1]
			local tweet_text=content[4]
			local tweet_id=content[2]
			if config.user_map[key]==nil then
				config.user_map[key]={}
				config.key_list[#config.key_list+1]=key
			end
			config.index2tweettext[#config.index2tweettext+1]=tweet_text
			table.insert(config.user_map[key],#config.index2tweettext)
			config.index2tweetid[#config.index2tweetid+1]=tweet_id
			config.tweet2index[tweet_id]=#config.index2tweetid
			-- Update the word model size
			config.word_model_size=config.word_model_size+((#utils.getNgrams(tweet_text,1,config.word_model_pad))-config.wwin+1)
		end
	end
	-- Update the tweet model size
	config.tweet_model_size=0
	config.tweet_model_pad=((config.twin-1)/2);
	for _,tweet_list in pairs(config.user_map) do
		config.tweet_model_size=config.tweet_model_size+((#tweet_list)-config.twin+1)
	end
	print(string.format('Word_Model #Iterations = %d; Tweet_Model #Iterations = %d;',config.word_model_size,config.tweet_model_size))
end

-- Function to sample image contexts
function utils.sample_image_contexts(config,context)
	local contexts=torch.Tensor(1+config.neg_samples,4096)
	contexts[1]=config.img_tensors[context]
	local i=0
	while i<config.neg_samples do
		neg_context=torch.random(#config.index2img)
		if context~=neg_context then
			contexts[i+2]=config.img_tensors[neg_context]
			i=i+1
		end
	end
	return contexts
end

-- Function to load image tensors
function utils.loadImageTensors(config)
	-- Load the image tensors
	config.img_tensors={}
	config.index2img={}
	config.img2index={}
	print('Loading image tensors...')
	start=sys.clock()
	count=0
	for line in io.lines(config.img_feat_file) do
		local content=utils.splitByChar(line,'\t')
		local imgId=content[1]
		config.index2img[#config.index2img+1]=imgId
		config.img2index[imgId]=#config.index2img
		local tensor=torch.Tensor(#content-1)
		for i=1,#content-1 do
			tensor[i]=tonumber(content[i+1])
		end
		table.insert(config.img_tensors,tensor)
		count=count+1
	end
	print(string.format("%d image features loaded. Done in %.2f seconds.",#config.img_tensors,sys.clock()-start))

	-- Load the image batches
	print('Loading image batches...')
	start=sys.clock()
	config.image_context_tensors={}
	config.image_target_tensors={}
	config.img_model_label_tensor=torch.zeros(1+config.neg_samples); config.img_model_label_tensor[1]=1;
	if config.gpu==1 then config.img_model_label_tensor=config.img_model_label_tensor:cuda() end
	local fptr=io.open('vis-tweets','w')
	local vt,nvt=0,0
	for line in io.lines(config.train_file) do
		local content=utils.splitByChar(line,'\t')
		if #content>=5 then
			-- considering tweets with images only
			for i=5,#content do
				local imgUrl=content[i]
				local idx=config.img2index[imgUrl]
				if idx~=nil then
					local tweetId=content[2]
					if config.model_type=='t2v-v-naive' or (config.model_type=='t2v-v-smart' and utils.isVisualTweet(config,config.tweet2index[tweetId],idx)==true) then
						local context=utils.sample_image_contexts(config,idx)
						local target=torch.IntTensor{config.tweet2index[tweetId]}
						if config.gpu==1 then
							context=context:cuda()
							target=target:cuda()
						end
						table.insert(config.image_context_tensors,context)
						table.insert(config.image_target_tensors,target)
						fptr:write(imgUrl..'\n')
						vt=vt+1
					else
						nvt=nvt+1
					end					
				end				
			end
			if (vt+nvt)%50==0 then
				collectgarbage()
			end
			if (vt+nvt)%2==0 then
				xlua.progress((vt+nvt),#config.img_tensors)
			end
		end
	end
	fptr:close()
	config.im_size=#config.image_target_tensors
	xlua.progress((vt+nvt),#config.img_tensors)
	print(string.format("Found Visual=%d & Non-Visual=%d image batches.",vt,nvt))
	print(string.format("Done in %.2f seconds.",sys.clock()-start))

	-- Clean the memory
	config.img_tensors=nil
	collectgarbage()
end

-- Function to check if the tweet is visual or not
function utils.isVisualTweet(config,tweetId,imgId)
	-- Find K-nearest image-tweets
	if config.image_tweets==nil then
		utils.populateImageTweets(config)
	end
	local neighbors=utils.getKNearestTweets(config,tweetId)

	-- Compute the dispersion and median score
	local score,median=utils.computeDispersionScore(config,neighbors,imgId)
	if score<median then
		return true
	end
	return false
end

-- Function to compute image dispersion score
function utils.computeDispersionScore(config,neighbors,targetImageId)
	local target_img_feat=config.img_tensors[targetImageId]
	local score=0
	local median=-1
	for i,content in ipairs(neighbors) do
		local sim=(1-nn.CosineDistance():forward({config.img_tensors[content],target_img_feat})[1])
		score=score+sim
		if i==torch.ceil(#neighbors/2) then
			median=sim
		end
	end
	score=score/#neighbors
	return score,median
end

-- Function to find k-nearest tweets by textual similarity (cosine)
function utils.getKNearestTweets(config,tweetId)
	local tensor=torch.Tensor(#config.image_tweets-1) -- exclude the target tweet
	local cur_tweet=config.tweet_vecs.weight[tweetId]
	-- Compute the cosine similarity
	local seqList={}
	for i,content in ipairs(config.image_tweets) do
		local tweetSeqNo=content[1]
		if tweetSeqNo~=tweetId then
			table.insert(seqList,content[2])
			tensor[#seqList]=config.distance:forward({config.tweet_vecs.weight[tweetSeqNo],cur_tweet})[1]
		end
	end
	-- Find the k-nearest tweets
	local result={}
	local score,order=torch.sort(tensor,true)  -- Setting 'true' to sort in descending order.
	for i=1,config.neighbors do
		table.insert(result,seqList[order[i]])
	end
	seqList=nil
	tensor=nil
	return result
end

-- Function to populate image tweets
function utils.populateImageTweets(config)
	config.image_tweets={}
	for line in io.lines(config.train_file) do
		local content=utils.splitByChar(line,'\t')
		if #content>=5 then
			-- considering tweets with images only
			local imgUrl=content[5]
			local idx=config.img2index[imgUrl]
			if idx~=nil then
				table.insert(config.image_tweets,{config.tweet2index[content[2]],idx})
			end
		end
	end
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