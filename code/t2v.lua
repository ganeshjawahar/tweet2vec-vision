--[[

Class for Tweet2Vec. 

--]]

local Tweet2Vec=torch.class("Tweet2Vec")
local utils=require 'utils'

-- Lua Constructor
function Tweet2Vec:__init(config)
	-- data	
	self.data_dir=config.data_dir
	self.train_file=self.data_dir..'/dataset.txt'
	self.img_feat_file=self.data_dir..'/imgfeat.tsv'
	self.out_file=config.out_file
	self.decay=config.decay

	-- model params (general)
	self.wdim=config.wdim
	self.wwin=config.wwin
	self.twin=config.twin
	self.min_freq=config.min_freq
	self.pad_tweet=config.pad_tweet
	self.model_type=config.model_type
	self.is_center_target=config.is_center_target
	self.print_params=config.print_params
	self.neg_samples=config.neg_samples
	self.neighbors=config.neighbors
	self.start_epoch=config.start_epoch

	-- optimization
	self.learning_rate=config.learning_rate
	self.batch_size=config.batch_size
	self.max_epochs=config.max_epochs
	self.reg=config.reg
	self.grad_clip=config.grad_clip
	self.pre_train=config.pre_train
	self.softmaxtree=config.softmaxtree

	-- GPU/CPU
	self.gpu=config.gpu

	-- Maps
	self.vocab={} -- word frequency map
    self.index2word={}
    self.word2index={}

	-- Build vocabulary
	utils.buildVocab(self)

	-- Load data set into memory
	utils.loadDataToMemory(self)

	-- build the net
    self:build_model()

	-- Initialize word vectors with pre-trained word embeddings
	if self.pre_train==1 then
		utils.initWordWeights(self,config.glove_dir..'glove.twitter.27B.'..config.wdim..'d.txt.gz')
	end

    if self.gpu==1 then
    	-- Ship stuffs to CUDA
    	self:cuda()
    end
end

-- Function to train the model
function Tweet2Vec:train()
	print('Training...')
	local start=sys.clock()

	-- Word model params & trainer
	local w_cur_batch_row=0
	self.w_optim_state={learningRate=self.learning_rate,alpha=self.decay}
	self.w_params,self.w_grad_params=self.wordModel:getParameters()
	self.w_feval=function(x)
		-- Get new params
		self.w_params:copy(x)

		-- Reset gradients
		self.w_grad_params:zero()

		-- loss is average of all criterions
		local loss=0
		local count=0
		for i=1,self.w_cur_batch_size do
			-- estimate f
			local input=nil
			if self.softmaxtree==1 then
				input={{self.word_context_tensors[i],self.tweet_target_tensors[i]},self.word_target_tensors[i]}
			else
				input={self.word_context_tensors[i],self.tweet_target_tensors[i]}
			end
			local output=self.wordModel:forward(input)
			local lb=self.word_target_tensors[i]
			local err=self.criterion:forward(output,lb)
			loss=loss+err

			-- estimate df/dW
			local bk=self.criterion:backward(output,lb)
			self.wordModel:backward(input,bk)
			count=count+1
		end
		loss=loss/count
		self.w_grad_params:div(count)

		--regularization
  		loss=loss+0.5*self.reg*self.w_params:norm()^2

		return loss,self.w_grad_params
	end

	-- Tweet model params & trainer
	local t_cur_batch_row=0
	self.t_optim_state={learningRate=self.learning_rate,alpha=self.decay}
	self.t_params,self.t_grad_params=self.tweetModel:getParameters()
	self.t_feval=function(x)
		-- Get new params
		self.t_params:copy(x)

		-- Reset gradients
		self.t_grad_params:zero()

		-- loss is average of all criterions
		local loss=0
		local count=0
		for i=1,self.t_cur_batch_size do
			local input=self.tweet_context_tensors[i]
			local lb=self.tweet_target_tensors[i]
			local output1=self.tweetModel1:forward(input)
			local output2=self.weightPredModel:forward(input)
			for row=1,(#output2)[1] do
				output1[row]=output1[row]:mul(output2[row]*(self.twin-1))
			end
			local output=nil
			if self.softmaxtree==1 then
				output=self.tweetModel2:forward({output1,lb})
			else
				output=self.tweetModel2:forward(output1)
			end

			local err=self.criterion:forward(output,lb)
			loss=loss+err

			-- estimate df/dW
			local bk=self.criterion:backward(output,lb)
			local grad_input=self.tweetModel2:backward({output1,lb},bk)
			grad_input=grad_input[1]
			grad_input=grad_input:contiguous()
			self.tweetModel1:backward(input,grad_input)
			self.weightPredModel:backward(input,grad_input)
			count=count+1
		end

		loss=loss/count
		self.t_grad_params:div(count)

		-- If the gradients explode, scale down the gradients
		if self.t_grad_params:norm()>=self.grad_clip then
			self.t_grad_params:mul(self.grad_clip/self.t_grad_params:norm())
		end
		--self.t_grad_params:clamp(-self.grad_clip,self.grad_clip)

		--regularization
		loss=loss+0.5*self.reg*self.t_grad_params:norm()^2

		return loss,self.t_grad_params
	end

	if self.model_type=='t2v-v-naive' or self.model_type=='t2v-v-smart' then
		-- Tweet image model params & trainer
		local i_cur_batch_row=0
		self.i_optim_state={learningRate=self.learning_rate,alpha=self.decay}
		self.i_params,self.i_grad_params=self.imgModel:getParameters()
		self.i_feval=function(x)
			-- Get new params
			self.i_params:copy(x)

			-- Reset gradients
			self.i_grad_params:zero()

			-- loss is average of all criterions
			local loss=0
			local last_sample=math.min(self.i_cur_batch+self.batch_size-1,self.im_size)
			local count=0
			for i=self.i_cur_batch,last_sample do
				-- estimate f
				local input={self.image_context_tensors[i],self.image_target_tensors[i]}
				local output=self.imgModel:forward(input)
				local err=self.criterion2:forward(output,self.img_model_label_tensor)
				loss=loss+err

				-- estimate df/dW
				local bk=self.criterion2:backward(output,self.img_model_label_tensor)
				self.imgModel:backward(input,bk)
				count=count+1
			end

			loss=loss/count
			self.i_grad_params:div(count)

			-- If the gradients explode, scale down the gradients
			if self.t_grad_params:norm()>=self.grad_clip then
				self.t_grad_params:mul(self.grad_clip/self.t_grad_params:norm())
			end
			--self.i_grad_params:clamp(-self.grad_clip,self.grad_clip)

			--regularization
			loss=loss+0.5*self.reg*self.i_grad_params:norm()^2

			return loss,self.i_grad_params
		end
	end
	
	for epoch=1,self.max_epochs do
		local epoch_start=sys.clock()
		local w_epoch_loss=0
		local w_epoch_iteration=0
		local t_epoch_loss=0
		local t_epoch_iteration=0
		local i_epoch_loss=0
		local i_epoch_iteration=0

		--[[
		print('Modeling word likelihood...')
		local total=#self.key_list
		local indices=torch.randperm(total)
		local row=0
		for index=1,total do
			local key=self.key_list[indices[index]
			local data=self.user_map[key]
			for tweet_index=1,#data do
				local t_id=data[tweet_index]
				local tweet_text=self.index2tweettext[t_id]
				local windows=utils.getWordWindows(tweet_text,self.word_model_pad,self.wwin,self.word2index,self.is_center_target)
				for wi,window in ipairs(windows) do
					self.word_context_tensors[w_cur_batch_row+1]=window[1]
					self.tweet_target_tensors[w_cur_batch_row+1]=t_id
					self.word_target_tensors[w_cur_batch_row+1]=window[2]
					w_cur_batch_row=w_cur_batch_row+1
					if w_cur_batch_row==self.batch_size then
						self.w_cur_batch_size=w_cur_batch_row
						local _,loss=optim.sgd(self.w_feval,self.w_params,self.w_optim_state)
						w_epoch_loss=w_epoch_loss+loss[1]
						w_epoch_iteration=w_epoch_iteration+1
						w_cur_batch_row=0
					end			
					row=row+1
					if row%200==0 then
						xlua.progress(row,self.word_model_size)
					end
				end
			end
		end
		if w_cur_batch_row~=0 then
			self.w_cur_batch_size=w_cur_batch_row
			local _,loss=optim.sgd(self.w_feval,self.w_params,self.w_optim_state)
			w_epoch_loss=w_epoch_loss+loss[1]
			w_epoch_iteration=w_epoch_iteration+1
		end
		xlua.progress(self.word_model_size,self.word_model_size)
		print(string.format("Word loss=%f\n",(w_epoch_loss/w_epoch_iteration)))

		-- Modeling tweet likelihood
		print('Modeling tweet likelihood...')
		indices=torch.randperm(total)
		row=0
		for index=1,total do
			local key=self.key_list[indices[index]			
			local data=self.user_map[key]
			local windows=utils.getTweetWindows(data,self.twin,self.corpus_size,1)
			for wi,window in ipairs(windows) do
				self.tweet_context_tensors[t_cur_batch_row+1]=window[1]
				self.tweet_target_tensors[t_cur_batch_row+1]=window[2]
				t_cur_batch_row=t_cur_batch_row+1
				if t_cur_batch_row==self.batch_size then
					self.t_cur_batch_size=t_cur_batch_row
					local _,loss=optim.sgd(self.t_feval,self.t_params,self.t_optim_state)
					t_epoch_loss=t_epoch_loss+loss[1]
					t_epoch_iteration=t_epoch_iteration+1
					t_cur_batch_row=0
				end
				row=row+1
				if row%200==0 then
					xlua.progress(row,self.tweet_model_size)
				end
			end
		end
		if t_cur_batch_row~=0 then
			self.t_cur_batch_size=t_cur_batch_row
			local _,loss=optim.sgd(self.t_feval,self.t_params,self.t_optim_state)
			t_epoch_loss=t_epoch_loss+loss[1]
			t_epoch_iteration=t_epoch_iteration+1
		end
		xlua.progress(self.tweet_model_size,self.tweet_model_size)
		print(string.format("Tweet textual loss=%f\n",(t_epoch_loss/t_epoch_iteration)))
		]]--

		if self.model_type=='t2v-v-naive' or self.model_type=='t2v-v-smart' then
			-- Modeling image features
			if epoch>=self.start_epoch then
				-- Load the image batches, if not done before
				if epoch==self.start_epoch then
					utils.loadImageTensors(self)
				end

				print('Modeling image features...')
				xlua.progress(1,self.im_size)
				for index=1,self.im_size,self.batch_size do
					if index%2==0 then
						xlua.progress(index*self.batch_size,self.im_size)
					end
					self.i_cur_batch=index
					local _,loss=optim.sgd(self.i_feval,self.i_params,self.i_optim_state)
					i_epoch_loss=i_epoch_loss+loss[1]
					i_epoch_iteration=i_epoch_iteration+1
				end
				xlua.progress(self.im_size,self.im_size)
				print(string.format("Tweet visual loss=%f\n",(i_epoch_loss/i_epoch_iteration)))
			end
		end
		print(string.format("Epoch %d done in %.2f minutes.\n\n",epoch,((sys.clock()-epoch_start)/60)))
	end
	print(string.format("Training done in %.2f minutes.",((sys.clock()-start)/60)))
end

-- Function to compute dev and test feature vectors
function Tweet2Vec:evaluate()
	-- Freeze updating word vectors.
	self.word_vecs.accGradParameters=function() end
	utils.saveEmbeddings('word',self.word_vecs.weight,self.index2word,'word_feat.txt')
	local start=sys.clock()
	utils.saveTweetEmbeddings(self.tweet_vecs.weight,'train_feat.txt')
	print(string.format("Train. rep saved in %.2f minutes.",((sys.clock()-start)/60)))
end

-- Function to build the model
function Tweet2Vec:build_model()
	self.corpus_size=self.corpus_size+self.twin-1 -- Account for temporal pads
	self.word_vecs=utils.getLookupTable(self.vocab_size,self.wdim,-0.05,0.05)
	self.tweet_vecs=utils.getLookupTable(self.corpus_size,self.wdim,-0.05,0.05)
	self.tweet_target_tensors=torch.IntTensor(self.batch_size,1)
	self.tweet_context_tensors=torch.IntTensor(self.batch_size,self.twin-1)
	self.word_target_tensors=torch.IntTensor(self.batch_size,1)
	self.word_context_tensors=torch.IntTensor(self.batch_size,self.wwin-1)

	-- Create the word model
	self.wordModelPart=nn.Sequential()
	self.wordModelPart:add(nn.ParallelTable())
	self.wordModelPart.modules[1]:add(self.word_vecs)
	self.wordModelPart.modules[1]:add(self.tweet_vecs)
	self.wordModelPart:add(nn.JoinTable(1))
	self.wordModelPart:add(nn.Mean())
	if self.softmaxtree==1 then
		self.wordModel=nn.Sequential()
		self.wordModel:add(nn.ParallelTable())
		self.wordModelPart:add(nn.Reshape(1,self.wdim))
		self.wordModel.modules[1]:add(self.wordModelPart)		
		self.wordModel.modules[1]:add(nn.Identity())
		-- Create the softmax tree layer
		tree,root=utils.create_frequency_tree(utils.create_word_map(self.vocab,self.index2word))
		self.wordModel:add(nn.SoftMaxTree(self.wdim,tree,root))		
	else
		self.wordModel=self.wordModelPart
		self.wordModel:add(nn.Linear(self.wdim,self.vocab_size))
		self.wordModel:add(nn.LogSoftMax())
	end	

	-- Create the adaptive weight prediction model
	self.weightPredModel=nn.Sequential()
	local tweet_vecs_clone_1=self.tweet_vecs:clone("weight","bias","gradWeight","gradBias")
	self.weightPredModel:add(tweet_vecs_clone_1)
	self.weightPredModel:add(nn.View((self.twin-1)*self.wdim))
	self.weightCompositionLayer=nn.Linear((self.twin-1)*self.wdim,self.twin-1)
	self.weightPredModel:add(self.weightCompositionLayer)
	self.weightPredModel:add(nn.SoftMax())

	-- Create the tweet part-1 model
	self.tweetModel1=nn.Sequential()
	local tweet_vecs_clone_2=self.tweet_vecs:clone("weight","bias","gradWeight","gradBias")
	self.tweetModel1:add(tweet_vecs_clone_2)

	-- Create the tweet part-2 model
	self.tweetModel2Part=nn.Sequential()
	self.tweetModel2Part:add(nn.Identity())
	self.tweetModel2Part:add(nn.Mean())
	if self.softmaxtree==1 then
		self.tweetModel2=nn.Sequential()
		self.tweetModel2:add(nn.ParallelTable())
		self.tweetModel2Part:add(nn.Reshape(1,self.wdim))
		self.tweetModel2.modules[1]:add(self.tweetModel2Part)		
		self.tweetModel2.modules[1]:add(nn.Identity())
		-- Create the softmax tree layer
		tree,root=utils.create_frequency_tree(utils.create_tweet_map(self.corpus_size))
		self.tweetModel2:add(nn.SoftMaxTree(self.wdim,tree,root))		
	else
		self.tweetModel2=self.tweetModel2Part
		self.tweetModel2:add(nn.Linear(self.wdim,self.corpus_size))
		self.tweetModel2:add(nn.LogSoftMax())
	end

	self.tweetModel=nn.Parallel():add(self.weightPredModel):add(self.tweetModel1):add(self.tweetModel2)

	if self.model_type=='t2v-v-naive' or self.model_type=='t2v-v-smart' then
		self.targetModel=nn.Sequential()		
		local tweet_vecs_clone_3=self.tweet_vecs:clone("weight","bias","gradWeight","gradBias")
		self.targetModel:add(tweet_vecs_clone_3)
		self.targetModel:add(nn.Linear(self.wdim,4096))		
		self.imgModel=nn.Sequential()
		self.imgModel:add(nn.ParallelTable())
		self.imgModel.modules[1]:add(nn.Identity())
		self.imgModel.modules[1]:add(self.targetModel)
		self.imgModel:add(nn.MM(false,true))
		self.imgModel:add(nn.Sigmoid())
		self.criterion2=nn.BCECriterion()
		self.distance=nn.CosineDistance()
	end

	-- Define the criterion
	if self.softmaxtree==1 then
		self.criterion=nn.TreeNLLCriterion()
	else	
		self.criterion=nn.ClassNLLCriterion()
	end
end

-- Function to ship stuffs to GPU
function Tweet2Vec:cuda()
	require 'cunn'
	require 'cunnx'
	require 'cutorch'
	self.criterion=self.criterion:cuda()
	self.tweet_target_tensors=self.tweet_target_tensors:cuda()
	self.tweet_context_tensors=self.tweet_context_tensors:cuda()
	self.word_target_tensors=self.word_target_tensors:cuda()
	self.word_context_tensors=self.word_context_tensors:cuda()
	self.wordModel=self.wordModel:cuda()
	self.weightPredModel=self.weightPredModel:cuda()
	self.tweetModel1=self.tweetModel1:cuda()
	self.tweetModel2=self.tweetModel2:cuda()
	if self.model_type=='t2v-v-naive' or self.model_type=='t2v-v-smart' then
		self.imgModel=self.imgModel:cuda()		
		self.criterion2=self.criterion2:cuda()
		self.distance=self.distance:cuda()
	end
end