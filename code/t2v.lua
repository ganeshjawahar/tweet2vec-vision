--[[

Class for Tweet2Vec. 

--]]

local Tweet2Vec=torch.class("Tweet2Vec")
local utils=require 'utils'

-- Lua Constructor
function Tweet2Vec:__init(config)
	-- data	
	self.data_dir=config.data_dir
	self.train_file=self.data_dir..'/dataset_small.txt'
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

	-- Load train set into memory
	utils.loadTensorsToMemory(self)

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
	local pad=0
	if self.pad_tweet==true then pad=((self.wwin-1)/2) end

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
		local last_sample=math.min(self.w_cur_batch+self.batch_size-1,self.wm_size)
		local count=0
		for i=self.w_cur_batch,last_sample do
			-- estimate f
			local input=nil
			if self.softmaxtree==1 then
				input={{self.word_model_word_context[i],self.word_model_tweet_target[i]},self.word_model_word_target[i]}
			else
				input={self.word_model_word_context[i],self.word_model_tweet_target[i]}
			end
			local output=self.wordModel:forward(input)
			local lb=self.word_model_word_target[i]
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
		local last_sample=math.min(self.t_cur_batch+self.batch_size-1,self.tm_size)
		local count=0
		for i=self.t_cur_batch,last_sample do
			local input=self.tweet_model_context[i]
			local lb=self.tweet_model_target[i]
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

		self.t_grad_params:clamp(-self.grad_clip,self.grad_clip)

		--regularization
		loss=loss+0.5*self.reg*self.t_grad_params:norm()^2

		return loss,self.t_grad_params
	end

	-- Tweet image model params & trainer
	local i_cur_batch_row=0
	self.i_optim_state={learningRate=self.learning_rate,alpha=self.decay}
	self.i_params,self.i_grad_params=self.tweetModel:getParameters()
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

		self.i_grad_params:clamp(-self.grad_clip,self.grad_clip)

		--regularization
		loss=loss+0.5*self.reg*self.i_grad_params:norm()^2

		return loss,self.i_grad_params
	end

	for epoch=1,self.max_epochs do
		local epoch_start=sys.clock()
		local w_epoch_loss=0
		local w_epoch_iteration=0
		local t_epoch_loss=0
		local t_epoch_iteration=0
		local i_epoch_loss=0
		local i_epoch_iteration=0

		-- Modeling word likelihood
		print('Modeling word likelihood...')
		for index=1,self.wm_size,self.batch_size do
			xlua.progress(index,self.wm_size)
			self.w_cur_batch=index
			local _,loss=optim.sgd(self.w_feval,self.w_params,self.w_optim_state)
			w_epoch_loss=w_epoch_loss+loss[1]
			w_epoch_iteration=w_epoch_iteration+1
		end
		xlua.progress(self.wm_size,self.wm_size)
		print(string.format("Word loss=%f\n",(w_epoch_loss/w_epoch_iteration)))

		-- Modeling tweet likelihood
		print('Modeling tweet likelihood...')
		for index=1,self.tm_size,self.batch_size do
			xlua.progress(index,self.tm_size)
			self.t_cur_batch=index
			local _,loss=optim.sgd(self.t_feval,self.t_params,self.t_optim_state)
			t_epoch_loss=t_epoch_loss+loss[1]
			t_epoch_iteration=t_epoch_iteration+1
		end
		xlua.progress(self.tm_size,self.tm_size)
		print(string.format("Tweet textual loss=%f\n",(t_epoch_loss/t_epoch_iteration)))

		if epoch>=self.start_epoch then
			-- Load the image batches, if not done before
			if epoch==self.start_epoch then
				utils.loadImageTensors(self)
			end

			-- Modeling image features
			print('Modeling image features...')
			for index=1,self.im_size,self.batch_size do
				xlua.progress(index,self.im_size)
				self.i_cur_batch=index
				local _,loss=optim.sgd(self.i_feval,self.i_params,self.i_optim_state)
				i_epoch_loss=i_epoch_loss+loss[1]
				i_epoch_iteration=i_epoch_iteration+1
			end
			xlua.progress(self.im_size,self.im_size)
			print(string.format("Tweet visual loss=%f\n",(i_epoch_loss/i_epoch_iteration)))
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
	local tweet_vecs_clone_1=self.tweet_vecs:clone()
	self.tweet_vecs:share(tweet_vecs_clone_1,"weight","bias","gradWeight","gradBias")
	self.weightPredModel:add(tweet_vecs_clone_1)
	self.weightPredModel:add(nn.View((self.twin-1)*self.wdim))
	self.weightCompositionLayer=nn.Linear((self.twin-1)*self.wdim,self.twin-1)
	self.weightPredModel:add(self.weightCompositionLayer)
	self.weightPredModel:add(nn.SoftMax())

	-- Create the tweet part-1 model
	self.tweetModel1=nn.Sequential()
	local tweet_vecs_clone_2=self.tweet_vecs:clone()
	self.tweet_vecs:share(tweet_vecs_clone_2,"weight","bias","gradWeight","gradBias")
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
		local tweet_vecs_clone_3=self.tweet_vecs:clone()
		self.tweet_vecs:share(tweet_vecs_clone_3,"weight","bias","gradWeight","gradBias")
		self.targetModel:add(tweet_vecs_clone_3)
		self.targetModel:add(nn.Linear(self.wdim,4096))		
		self.imgModel=nn.Sequential()
		self.imgModel:add(nn.ParallelTable())
		self.imgModel.modules[1]:add(nn.Identity())
		self.imgModel.modules[1]:add(self.targetModel)
		self.imgModel:add(nn.MM(false,true))
		self.imgModel:add(nn.Sigmoid())
		self.tweetModel:add(imgModel)
		self.criterion2=nn.BCECriterion()
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
	self.word_model_word_context=self.word_model_word_context:cuda()
	self.word_model_tweet_target=self.word_model_tweet_target:cuda()
	self.word_model_word_target=self.word_model_word_target:cuda()
	self.tweet_model_context=self.tweet_model_context:cuda()
	self.tweet_model_target=self.tweet_model_target:cuda()
	self.wordModel=self.wordModel:cuda()
	self.weightPredModel=self.weightPredModel:cuda()
	self.tweetModel1=self.tweetModel1:cuda()
	self.tweetModel2=self.tweetModel2:cuda()
	if self.model_type=='t2v-v-naive' or self.model_type=='t2v-v-smart' then
		self.imgModel=self.imgModel:cuda()		
		self.criterion2=self.criterion2:cuda()
	end
end