--[[

Tweet2Vec: Learning Tweet Representations by bridging language and vision gap.

]]--

require 'torch'
require 'io'
require 'nn'
require 'nnx'
require 'sys'
require 'optim'
require 'os'
require 'xlua'
require 'lfs'
require 'cunn'
require 'cutorch'
include('t2v.lua')

cmd = torch.CmdLine()
cmd:text()
cmd:text('Tweet2Vec: Learning Tweet Representations by bridging language and vision gap')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','../data/','Directory for accessing the user data.')
cmd:option('-glove_dir','/home/ganesh/Desktop/wsupe/data/Glove/','Directory for accesssing the pre-trained glove word embeddings')
cmd:option('-out_file','embeddings.txt','output file name')
cmd:option('-to_lower',1,'change the case of word to lower case')
-- model params (general)
cmd:option('-wdim',100,'dimensionality of word embeddings')
cmd:option('-wwin',19,'defines context words in a document for word modeling')
cmd:option('-twin',13,'defines context tweets in a stream for tweet modeling')
cmd:option('-min_freq',5,'words that occur less than <int> times will not be taken for training')
cmd:option('-pad_tweet',1,'should we need to pad the tweet ?')
cmd:option('-is_center_target',1,'model center element based on its surrounding words?')
cmd:option('-pre_train',1,'initialize word embeddings with pre-trained vectors?')
cmd:option('-model_type','t2v-v-smart','t2v or t2v-v-naive or t2v-v-smart')
cmd:option('-neg_samples',10,'while modeling vision features how many negative samples must be considered?')
cmd:option('-neighbors',10,'# nearest neighbors to be considered for compute the visual score')
cmd:option('-start_epoch',1,'epoch after which the model starts learning from visual features')
-- optimization
cmd:option('-learning_rate',0.01,'learning rate')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-batch_size',200,'number of sequences to train on in parallel')
cmd:option('-max_epochs',5,'number of full passes through the training data')
cmd:option('-reg',1e-4,'regularization parameter l2-norm')
cmd:option('-softmaxtree',1,'use SoftmaxTree instead of the inefficient (full) softmax')
-- GPU/CPU
cmd:option('-gpu',1,'1=use gpu; 0=use cpu;')
-- Book-keeping
cmd:option('-print_params',0,'output the parameters in the console. 0=dont print; 1=print;')
cmd:option('-decay',0.005,'decay rate')

-- parse input params
params=cmd:parse(arg)

if params.print_params==1 then
	-- output the parameters	
	for param, value in pairs(params) do
	    print(param ..' : '.. tostring(value))
	end
end

-- initialize the model
model=Tweet2Vec(params)

-- train the model
model:train()

-- evaluate the model
model:evaluate()