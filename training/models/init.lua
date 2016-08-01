--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Generic model creating code. For the specific ResNet model see
--  models/resnet.lua
--

require 'nn'
require 'cunn'
require 'cudnn'

local M = {}

function M.setup(opt, checkpoint)
   local model
   if checkpoint then
      local modelPath = paths.concat(opt.resume, checkpoint.modelFile)
      assert(paths.filep(modelPath), 'Saved model not found: ' .. modelPath)
      print('=> Resuming model from ' .. modelPath)
      model = torch.load(modelPath)
   elseif opt.retrain ~= 'none' then
      assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
      print('Loading model from file: ' .. opt.retrain)
      model = torch.load(opt.retrain)
   else
      print('=> Creating model from file: models/' .. opt.netType .. '.lua')
      model = require('models/' .. opt.netType)(opt)
   end

   -- First remove any DataParallelTable
   if torch.type(model) == 'nn.DataParallelTable' then
      model = model:get(1)
   end

   -- This is useful for fitting ResNet-50 on 4 GPUs, but requires that all
   -- containers override backwards to call backwards recursively on submodules
   if opt.shareGradInput then
      -- Share gradInput for memory efficient backprop
      local cache = {}
      model:apply(function(m)
         local moduleType = torch.type(m)
         if torch.isTensor(m.gradInput) and moduleType ~= 'nn.ConcatTable' then
            if cache[moduleType] == nil then
               cache[moduleType] = torch.CudaStorage(1)
            end
            m.gradInput = torch.CudaTensor(cache[moduleType], 1, 0)
         end
      end)
      for i, m in ipairs(model:findModules('nn.ConcatTable')) do
         if cache[i % 2] == nil then
            cache[i % 2] = torch.CudaStorage(1)
         end
         m.gradInput = torch.CudaTensor(cache[i % 2], 1, 0)
      end
   end

   -- For resetting the classifier when fine-tuning on a different Dataset
   if opt.resetClassifier and not checkpoint then
      print(' => Replacing classifier with ' .. opt.nClasses .. '-way classifier')

      local orig = model:get(#model.modules)
      assert(torch.type(orig) == 'nn.Linear',
         'expected last layer to be fully connected')
      
--       -- Add dropout after each block
--       local numDrop = {}
--       for i=1,  #model.modules do
-- 	print (torch.type(model:get(i)))
-- 	if torch.type(model:get(i)) == 'nn.Sequential' then
-- 	  print (i)
-- 	  table.insert(numDrop, i)
-- 	end
--       end
--       
--       for key,value in pairs(numDrop) do
-- 	print(key,value)
-- 	model:insert(nn.Dropout(0.400000):cuda(), value  + key)
--       end
      
--      -- Add dropout after each block
--       local numDrop = {}
--       local numDropIdx = {}
--       for i=1,  #model.modules do
-- 	print (torch.type(model:get(i)))
-- 	if torch.type(model:get(i)) == 'nn.Sequential' then
-- 	  local tModule = {}
-- 	  for j=1,  #model:get(i).modules do
-- 	    if torch.type(model:get(i):get(j)) == 'nn.Sequential' then
-- 	      table.insert(tModule, j)
-- 	    end
-- 	  end
-- 	  table.insert(numDropIdx, i)
-- 	  table.insert(numDrop, tModule)
-- 	end
--       end
--       
--       local nunSeq = 1
--       for key,value in pairs(numDropIdx) do
-- 	print (numDrop[value], nunSeq)
-- 	for idx,num in pairs(numDrop[nunSeq]) do
-- 	  model.modules[value]:insert(nn.Dropout(0.400000):cuda(), idx + num)
-- 	end
-- 	nunSeq = nunSeq + 1
--       end
	 
      local linear = nn.Linear(orig.weight:size(2), opt.nClasses)
      linear.bias:zero()

      model:remove(#model.modules)
      model:add(linear:cuda())
-- --       print (model)
--       
--       local function ConvInit(name)
--       for k,v in pairs(model:findModules(name)) do
--          local n = v.kW*v.kH*v.nOutputPlane
--          v.weight:normal(0,math.sqrt(2/n))
--          if cudnn.version >= 4000 then
--             v.bias = nil
--             v.gradBias = nil
--          else
--             v.bias:zero()
--          end
--       end
--      end
-- 
-- 
--       ConvInit('cudnn.SpatialConvolution')
--       ConvInit('nn.SpatialConvolution')
--     
--       for k,v in pairs(model:findModules('nn.Linear')) do
-- 	  v.bias:zero()
--       end
--       model:cuda()
-- 
--       if opt.cudnn == 'deterministic' then
-- 	  model:apply(function(m)
-- 	    if m.setMode then m:setMode(1,1,1) end
-- 	  end)
--       end
-- 
--       model:get(1).gradInput = nil
   
   end

   -- Set the CUDNN flags
   if opt.cudnn == 'fastest' then
      cudnn.fastest = true
      cudnn.benchmark = true
   elseif opt.cudnn == 'deterministic' then
      -- Use a deterministic convolution implementation
      model:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
   end

   -- Wrap the model with DataParallelTable, if using more than one GPU
   if opt.nGPU > 1 then
      local gpus = torch.range(1, opt.nGPU):totable()
      local fastest, benchmark = cudnn.fastest, cudnn.benchmark

      local dpt = nn.DataParallelTable(1, true, true)
         :add(model, gpus)
         :threads(function()
            local cudnn = require 'cudnn'
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
         end)
      dpt.gradInput = nil

      model = dpt:cuda()
   end

   local criterion = nn.CrossEntropyCriterion():cuda()
   -- local criterion = nn.ClassNLLCriterion():cuda()
   return model, criterion
end

return M
