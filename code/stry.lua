require 'torch'
require 'nn'
require 'nnx'
require 'dp'
require 'dpnn'
require 'rnn'

word_freq={}
word_freq[1]=15
word_freq[2]=25
word_freq[3]=10
word_freq[4]=5
word_freq[5]=32
word_freq[6]=22
word_freq[7]=100
word_freq[8]=52
word_freq[9]=150
word_freq[10]=215

function create_frequency_tree(word_freq)
	binSize = 100
	local wf = torch.IntTensor(word_freq)
	local vals, indices = wf:sort()
	local tree = {}
	local id = indices:size(1)
	function recursiveTree(indices)
		if indices:size(1) < binSize then
			id = id + 1
			tree[id] = indices
			return
		end
		local parents = {}
		for start=1,indices:size(1),binSize do
			local stop = math.min(indices:size(1), start+binSize-1)
			local bin = indices:narrow(1, start, stop-start+1)
			assert(bin:size(1) <= binSize)
			id = id + 1
			table.insert(parents, id)
			tree[id] = bin
		end
		recursiveTree(indices.new(parents))
	end
	recursiveTree(indices)
	return tree, id
end

lu=nn.LookupTable(10,5)
main=nn.Sequential()
main:add(nn.ParallelTable())
model=nn.Sequential()
model:add(lu)
model:add(nn.Mean())
model:add(nn.Reshape(1,5))
--pred=model:forward(torch.Tensor{1,2,3})
--print(pred)
tree,id=create_frequency_tree(word_freq)
softmax=nn.SoftMaxTree(5,tree,id)
main.modules[1]:add(model)
main.modules[1]:add(nn.Identity())
main:add(softmax)
pred=main:forward({torch.Tensor{1,2,3},torch.Tensor{4}})
criterion=nn.TreeNLLCriterion()
grad=criterion:backward(pred,torch.IntTensor{4})
main:backward({torch.Tensor{1,2,3},torch.IntTensor{4}},grad)

--[[
res=softmax:forward{pred,torch.IntTensor{4}}
criterion=nn.TreeNLLCriterion()
loss=criterion:forward(res,torch.IntTensor{4})
grad=criterion:backward(res,torch.IntTensor{4})
grad1=softmax:backward({pred,torch.IntTensor{4}},grad)
model:backward(torch.Tensor{1,2,3},grad1[1])

--sm:forward{lu.weight,torch.IntTensor{1}}
--model:add(nn.Sequencer(nn.SoftMaxTree(5,tree,id,false)))
--pred=model:forward({torch.Tensor{1,2,3}})
--print(pred)
--[[
input = torch.randn(5,10)
target = torch.IntTensor{20,24,27,10,12}
gradOutput = torch.randn(5)
root_id = 29
input_size = 10   
hierarchy = {
    [29]=torch.LongTensor{30,1,2}, [1]=torch.IntTensor{3,4,5}, 
    [2]=torch.IntTensor{6,7,8}, [3]=torch.IntTensor{9,10,11},
    [4]=torch.IntTensor{12,13,14}, [5]=torch.IntTensor{15,16,17},
    [6]=torch.IntTensor{18,19,20}, [7]=torch.IntTensor{21,22,23},
    [8]=torch.IntTensor{24,25,26,27,28}
 }
smt = nn.SoftMaxTree(input_size, hierarchy, root_id)
res=smt:forward{input, target}
criterion=nn.TreeNLLCriterion()
loss=criterion:forward(res,torch.IntTensor{10})
grad=criterion:backward(res,torch.IntTensor{10})
grad1,use=smt:backward({input,target},grad)
print(grad1)
]]--