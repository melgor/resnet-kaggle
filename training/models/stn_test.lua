-- @Author: melgor
-- @Date:   2016-04-27 18:44:15
-- @Last Modified by:   melgor
-- @Last Modified time: 2016-04-27 19:13:23

require 'nn'
require 'cunn'
require 'stn' 
require 'cudnn'



local function ConvInit(net, name)
  for k,v in pairs(net:findModules(name)) do
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


function createSTNModule( )
  local rot = true
  local sca = true
  local tra = true


  local localization_network = nn.Sequential()
  localization_network:add(cudnn.SpatialConvolution(3,64,9,9,4,4,3,3))
  localization_network:add(cudnn.ReLU())  
  localization_network:add(cudnn.SpatialConvolution(64,64,7,7,3,3,1,1))
  localization_network:add(cudnn.ReLU())
  localization_network:add(cudnn.SpatialConvolution(64,64,5,5,2,2,1,1))
  localization_network:add(cudnn.ReLU())
  localization_network:add(cudnn.SpatialConvolution(64,64,5,5,2,2,1,1))
  localization_network:add(cudnn.ReLU())
  localization_network:add(nn.View(576))
  localization_network:add(nn.Linear(576,100))
  localization_network:add(cudnn.ReLU())
  ConvInit(localization_network, 'cudnn.SpatialConvolution')

  local init_bias = {}
  local nbr_params = 0
  if rot then
      nbr_params = nbr_params + 1
      init_bias[nbr_params] = 0
    end
  if sca then
    nbr_params = nbr_params + 1
    init_bias[nbr_params] = 1
  end
  if tra then
    nbr_params = nbr_params + 2
    init_bias[nbr_params-1] = 0
    init_bias[nbr_params] = 0
  end
  if nbr_params == 0 then
    -- fully parametrized case
    nbr_params = 6
    init_bias = {1,0,0,0,1,0}
  end
   
  localization_network:add(nn.Linear(100,nbr_params))     
  localization_network:get(12).weight:zero()
  localization_network:get(12).bias = torch.Tensor(init_bias)



  local ct = nn.ConcatTable()
  local branch1 = nn.Sequential()
  branch1:add(nn.Transpose({3,4},{2,4}))
  branch1:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
  local branch2 = nn.Sequential()
  branch2:add(localization_network)
  branch2:add(nn.AffineTransformMatrixGenerator(rot, sca, tra))  
  branch2:add(nn.AffineGridGeneratorBHWD(224, 224)) 
  branch2:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
  ct:add(branch1)
  ct:add(branch2)

  local st = nn.Sequential()
  st:add(ct)
  local sampler = nn.BilinearSamplerBHWD()
  sampler:type('torch.FloatTensor')
  sampler.type = function(type)
        return self
      end
  st:add(sampler)
  st:add(nn.Copy('torch.FloatTensor','torch.CudaTensor', true, true))
  st:add(nn.Transpose({2,4},{3,4}))


  st:cuda()

  return st
end