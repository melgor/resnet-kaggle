require 'xlua'
require 'optim'
require 'torchx'
require 'cunn'
require 'cutorch'
require 'nn'
require 'dataset'
require 'image'
require 'hdf5'

local nCrops = 1

opt = lapp[[
   -v,--valData               (default "val.txt")       path to txt file <path> <label>
   --save                     (default 'predictions.h5')         name of prediction
   --batchSize                (default 32)         name of prediction
]]

print(opt)
local name = 'aug'


local  data = ImageDataset(opt.valData, opt, name, 1)
data.preprocess_image = data:preprocess()

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


function testData(data, name)
  print('==>'.." testing")
  local N = 0
  local outputs = {}
  local target = {}
  local indices     = torch.linspace(1,data.targets:size(1), data.targets:size(1)):long():split(math.floor(opt.batchSize))
  for t,v in ipairs(indices) do
    xlua.progress(t, #indices)
    batch, tar = loadImages(v,data)
    for i=1,opt.batchSize do
      image.save( 'data/c' .. tar[i] ..  "/" .. name .. t .. "_" .. i .. "_img.jpg", batch[i])
    end
  end

end
-- testData(data, "first")
-- testData(data, "second")
-- testData(data, "third")
testData(data, "fourth")




