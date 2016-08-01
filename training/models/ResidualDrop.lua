require 'nn'
require 'cudnn'
require 'cunn'

local ResidualDrop, parent = torch.class('nn.ResidualDrop', 'nn.Container')

function ResidualDrop:__init(deathRate,shortcutType, nChannels, nOutChannels, stride, typeBlock)
    parent.__init(self)
    self.gradInput = torch.Tensor()
    self.gate = true
    self.train = true
    self.deathRate = deathRate
    self.shortcutType = shortcutType
    nOutChannels = nOutChannels or nChannels
    stride = stride or 1
    self:basicblock(nChannels, nOutChannels, stride)
    self.skip = self:shortcut(nChannels,nOutChannels,stride)

    self.modules = {self.net, self.skip}
end


function ResidualDrop:basicblock(nChannels, nOutChannels, stride)
    self.net = nn.Sequential()
    self.net:add(cudnn.SpatialConvolution(nChannels, nOutChannels, 3,3, stride,stride, 1,1))
    self.net:add(cudnn.SpatialBatchNormalization(nOutChannels))
    self.net:add(cudnn.ReLU(true))
    self.net:add(cudnn.SpatialConvolution(nOutChannels, nOutChannels,
                                      3,3, 1,1, 1,1))
    self.net:add(cudnn.SpatialBatchNormalization(nOutChannels))
end

function ResidualDrop:bottleneck(nChannels, nOutChannels, stride)
    self.net = nn.Sequential()
    self.net:add(cudnn.SpatialConvolution(nChannels,nOutChannels,1,1,1,1,0,0))
    self.net:add(cudnn.SpatialBatchNormalization(nOutChannels))
    self.net:add(cudnn.ReLU(true))
    self.net:add(cudnn.SpatialConvolution(nOutChannels,nOutChannels,3,3,stride,stride,1,1))
    self.net:add(cudnn.SpatialBatchNormalization(nOutChannels))
    self.net:add(cudnn.ReLU(true))
    self.net:add(cudnn.SpatialConvolution(nOutChannels,nOutChannels*4,1,1,1,1,0,0))
    self.net:add(cudnn.SpatialBatchNormalization(nOutChannels * 4))
end



-- The shortcut layer is either identity or 1x1 convolution
function ResidualDrop:shortcut(nInputPlane, nOutputPlane, stride)
  local useConv = self.shortcutType == 'C' or
      (self.shortcutType == 'B' and nInputPlane ~= nOutputPlane)
  if useConv then
      -- 1x1 convolution
      return nn.Sequential()
        :add(cudnn.SpatialConvolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
        :add(cudnn.SpatialBatchNormalization(nOutputPlane))
  elseif nInputPlane ~= nOutputPlane then
      -- Strided, zero-padded identity shortcut
      return nn.Sequential()
        :add(nn.SpatialAveragePooling(1, 1, stride, stride))
        :add(nn.Concat(2)
            :add(nn.Identity())
            :add(nn.MulConstant(0)))
  else
      return nn.Identity()
  end
end

function ResidualDrop:updateOutput(input)
    local skip_forward = self.skip:forward(input)
    self.output:resizeAs(skip_forward):copy(skip_forward)
    if self.train then
      if self.gate then -- only compute convolutional output when gate is open
        self.output:add(self.net:forward(input))
      end
    else
      self.output:add(self.net:forward(input):mul(1-self.deathRate))
    end
    return self.output
end

function ResidualDrop:updateGradInput(input, gradOutput)
   self.gradInput = self.gradInput or input.new()
   self.gradInput:resizeAs(input):copy(self.skip:updateGradInput(input, gradOutput))
   if self.gate then
      self.gradInput:add(self.net:updateGradInput(input, gradOutput))
   end
   return self.gradInput
end

function ResidualDrop:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if self.gate then
      self.net:accGradParameters(input, gradOutput, scale)
   end
end