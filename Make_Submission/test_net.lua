require 'xlua'
require 'optim'
require 'torchx'
require 'cunn'
require 'cutorch'
require 'nn'
require 'cudnn'
require 'dataset'
require 'image'
require 'hdf5'
require 'optim'
optnet = require 'optnet'
local nCrops = 1

opt = lapp[[
   -v,--valData               (default "val.txt")       path to txt file <path> <label>
   -b,--batchSize             (default 64)          batch size
   --model                    (default lenet)     trained model
   --manualSeed               (default 10)        seed to reproduce results
   --GPU                      (default 1)         GPU ID
   --aug                      (default 1)         get results by TTA, set number of augumentation
   --save                     (default 'predictions.h5')         name of prediction
]]
torch.manualSeed(opt.manualSeed)
cutorch.manualSeed(opt.manualSeed)
cutorch.setDevice(opt.GPU)
print(opt)
local name = 'val'
if opt.aug > 1 then
  name = 'aug'
end

local  data = ImageDataset(opt.valData,opt, name, opt.aug)
data.preprocess_image = data:preprocess()
local  model = torch.load(opt.model):clearState()
local criterion = nn.CrossEntropyCriterion():float()
local confusion =  optim.ConfusionMatrix(10)
-- Replace Averege Pollling 8x8-256 10x10-320
-- local num = 13
-- model:remove(num)
-- model:remove(num)
-- model:remove(num)
-- model:insert(cudnn.SpatialAveragePooling(8,8,1,1):cuda(),num)
-- model:cuda()

-- print(model:forward(torch.CudaTensor(1,3,256,256)):size())
-- print(model:forward(torch.CudaTensor(1,3,256,256)))

-- local num = 33
-- model:remove(num)
-- model:insert(cudnn.SpatialAveragePooling(16,16,1,1):cuda(),num)



-- optnet.optimizeMemory(model, torch.CudaTensor(10,3,224,224),'evaluation')

function loadImages(indices, data )
  local sz = indices:size(1)
  local batch, imageSize
  local target = torch.IntTensor(sz)
  for i, idx in ipairs(indices:totable()) do
    local sample = data:get(idx)
--     local input = data.preprocess_image(sample.input)
--     if not batch then
--       imageSize = input:size():totable()
--       if data.aug > 1 then table.remove(imageSize, 1) end
--       batch = torch.FloatTensor(sz, data.aug, table.unpack(imageSize))
--     end
--     batch[i]:copy(input)
    for j=1,data.aug do
      local input = data.preprocess_image(sample.input)
      if not batch then
        imageSize = input:size():totable()
        batch = torch.FloatTensor(sz * data.aug, table.unpack(imageSize))
      end
      batch[(i-1)*data.aug + j]:copy(input)
    end
    target[i] = sample.target
  end
--   return batch:view(sz * data.aug, table.unpack(imageSize)), target
  return batch, target
end
local softMaxLayer = nn.SoftMax():float()
local logSoftMaxLayer = nn.LogSoftMax():float()
local criterionLog = nn.ClassNLLCriterion():float()

local fulllyConv  = 16
function computeScore(output, target, aug)

  if aug > 1 then
    
      -- Sum over crops
      output = output:view(output:size(1) / aug, aug, output:size(2))
         --:exp()
         :sum(2):squeeze(2):div(aug)
   end
   -- Computes the top1 and top5 error rate
   local batchSize = output:size(1)

   local _ , predictions = output:float():sort(2, true) -- descending

   -- Find which predictions match the target
   local correct = predictions:eq(
      target:long():view(batchSize, 1):expandAs(output))

   -- Top-1 score
   local top1 = 1.0 - (correct:narrow(2, 1, 1):sum() / batchSize)
-- 
   return top1 * 100, output
end

function testData(data)
  -- disable flips, dropouts and batch normalization
  model:evaluate()
  print('==>'.." testing")
  local N = 0
  local outputs = {}
  local target = {}
  local indices     = torch.linspace(1,data.targets:size(1), data.targets:size(1)):long():split(math.floor(opt.batchSize / opt.aug))
  local loss2 = {}
  for t,v in ipairs(indices) do
    xlua.progress(t, #indices)
    batch, tar = loadImages(v,data)
    local out      = model:forward(batch:cuda()):float()
    local maxs, indices = torch.max(out, 2)
    print (logSoftMaxLayer:forward(out))
    print(tar)
    print(indices)
    print(logSoftMaxLayer:forward(out)[{{1},{10}}])
    table.insert(outputs,out)
    table.insert(target,tar)
    table.insert(loss2, criterion:forward(out, tar))
  end
  outputs            = torch.concat(outputs):float()
  target             = torch.concat(target):float()
  local top1, outSq  = computeScore(outputs, target, opt.aug)
  local loss         = criterion:forward(outSq, target)
  confusion:batchAdd(outSq, target)
  
  print(('Top1: %.3f Loss: %.6f'):format(top1, loss))
  confusion:updateValids()
  print(tostring(confusion))
  
  
  return outSq
end

local out = testData(data)

local myFile = hdf5.open(opt.save, 'w')
myFile:write('prediction',softMaxLayer:forward(out))
myFile:close()




