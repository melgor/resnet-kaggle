---
--  The Simple model with input 64x64 model definition
--

local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max =  cudnn.SpatialMaxPooling
local SBatchNorm = cudnn.SpatialBatchNormalization

local function createModel(opt)
    local model = nn.Sequential()
    model:add(Convolution(3,32,3,3,1,1,1,1))
    model:add(SBatchNorm(32))
    model:add(ReLU(true))
    model:add(Max(3,3,2,2,1,1))
    model:add(Convolution(32,64,3,3,1,1,1,1))
    model:add(SBatchNorm(64))
    model:add(ReLU(true))
    model:add(Max(3,3,2,2,1,1))
    model:add(Convolution(64,64,3,3,1,1,0,0))
    model:add(SBatchNorm(64))
    model:add(Max(3,3,2,2,1,1))
    model:add(Convolution(64,64,3,3,1,1,0,0))
    model:add(SBatchNorm(64))
    model:add(Avg(5, 5, 1, 1))
    model:add(nn.View(64):setNumInputDims(3))
    model:add(nn.Linear(64, 10))
  

   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('fbnn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   model:cuda()

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   model:get(1).gradInput = nil

   return model
end

return createModel
