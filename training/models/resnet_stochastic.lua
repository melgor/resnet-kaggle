
require 'nn'
require 'cudnn'
require 'cunn'


local Avg = cudnn.SpatialAveragePooling

---- Adds a residual block to the passed in model ----
function addResidualDrop(model, deathRate, shortcutType, nChannels, nOutChannels, stride)
   model:add(nn.ResidualDrop(deathRate, shortcutType, nChannels, nOutChannels, stride))
   model:add(cudnn.ReLU(true))
   return model
end

local function createModel(opt)
  local shortcutType = opt.shortcutType or 'B'
  local depth = opt.depth
  local deathRate = opt.deathRate or 0.5
  local iChannels
  
  -- Creates count residual blocks with specified number of features
  local function layer(model, featuresOut, count, stride)
      local s = nn.Sequential()
      local nInputPlane = iChannels
      iChannels = featuresOut
      for i=1,count do
         addResidualDrop(model, nil, shortcutType, nInputPlane, iChannels, i == 1 and stride or 1)
         nInputPlane = iChannels
      end
      return s
   end
   
  -- Configurations for ResNet:
  --  num. residual blocks, num features, residual block function
  local cfg = {
      [18]  = {{2, 2, 2, 2}, 512, basicblock},
      [34]  = {{3, 4, 6, 3}, 512, basicblock},
      [50]  = {{3, 4, 6, 3}, 2048, bottleneck},
      [101] = {{3, 4, 23, 3}, 2048, bottleneck},
      [152] = {{3, 8, 36, 3}, 2048, bottleneck},
  }
  iChannels = 64
  assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
  local def, nFeatures, block = table.unpack(cfg[depth])
   
   
  local model = nn.Sequential()
  ------> 3, 32,32
  model:add(cudnn.SpatialConvolution(3,64,7,7,2,2,3,3))
  model:add(cudnn.SpatialBatchNormalization(64))
  model:add(cudnn.ReLU(true))
  model:add(cudnn.SpatialMaxPooling(3,3,2,2,1,1))
  layer(model, 64,  def[1])
  layer(model, 128, def[2], 2)
  layer(model, 256, def[3], 2)
  layer(model, 512, def[4], 2)
  model:add(Avg(7, 7, 1, 1))
--model:add(nn.Dropout(0.5))
  model:add(nn.View(nFeatures):setNumInputDims(3))
  model:add(nn.Linear(nFeatures, 10))
  
    
  ---- Determines the position of all the residual blocks ----
  addtables = {}
  for i=1,model:size() do
      if tostring(model:get(i)) == 'nn.ResidualDrop' then addtables[#addtables+1] = i end
  end

  ---- Sets the deathRate (1 - survival probability) for all residual blocks  ----
  for i,block in ipairs(addtables) do
     model:get(block).deathRate = i / #addtables * deathRate
  end
    
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
  
  function openAllGates()
    for i,block in ipairs(addtables) do model:get(block).gate = true end
  end
  
  function closeGatesRandomly()
    for i,tb in ipairs(addtables) do
      if torch.rand(1)[1] < model:get(tb).deathRate then model:get(tb).gate = false end
    end
  end
  
  -- Set special methods for model train and evaulate
  model.openAllGates = openAllGates
  model.closeGatesRandomly = closeGatesRandomly
  
  return model  
end

return createModel    
