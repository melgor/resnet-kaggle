--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Image transforms for data augmentation and input normalization
--

require 'image'
local iproc = require 'iproc'
local M = {}

function M.Compose(transforms)
   return function(input)
      for _, transform in ipairs(transforms) do
         input = transform(input)
      end
      return input
   end
end

function M.ColorNormalize(meanstd)
   return function(img)
      img = img:clone()
      for i=1,3 do
         img[i]:add(-meanstd.mean[i])
--          img[i]:div(meanstd.std[i])
      end
      return img
   end
end

-- RGB to BGR and scale image to 0-255
function M.VGG_Preprocess()
  return function(img)
      local im2      = img:clone()
      im2[{1,{},{}}] = img[{3,{},{}}]
      im2[{3,{},{}}] = img[{1,{},{}}]
      im2:mul(255)
      return im2
  end
end

-- Scales the smaller edge to size
function M.Scale(size, interpolation)
   interpolation = interpolation or 'bicubic'
   return function(input)
      local w, h = input:size(3), input:size(2)
      if (w <= h and w == size) or (h <= w and h == size) then
         return input
      end
      if w < h then
         return image.scale(input, size, h/w * size, interpolation)
      else
         return image.scale(input, w/h * size, size, interpolation)
      end
   end
end

-- Crop to centered rectangle
function M.CenterCrop(size)
   return function(input)
      local w1 = math.ceil((input:size(3) - size)/2)
      local h1 = math.ceil((input:size(2) - size)/2)
      return image.crop(input, w1, h1, w1 + size, h1 + size) -- center patch
   end
end

-- Random crop form larger image with optional zero padding
function M.RandomCrop(size, padding)
   padding = padding or 0

   return function(input)
      if padding > 0 then
         local temp = input.new(3, input:size(2) + 2*padding, input:size(3) + 2*padding)
         temp:zero()
            :narrow(2, padding+1, input:size(2))
            :narrow(3, padding+1, input:size(3))
            :copy(input)
         input = temp
      end

      local w, h = input:size(3), input:size(2)
      if w == size and h == size then
         return input
      end

      local x1, y1 = torch.random(0, w - size), torch.random(0, h - size)
      local out = image.crop(input, x1, y1, x1 + size, y1 + size)
      assert(out:size(2) == size and out:size(3) == size, 'wrong crop size')
      return out
   end
end

-- Four corner patches and center crop from image and its horizontal reflection
function M.TenCrop(size)
   local centerCrop = M.CenterCrop(size)

   return function(input)
      local w, h = input:size(3), input:size(2)

      local output = {}
      for _, img in ipairs{input, image.hflip(input)} do
         table.insert(output, centerCrop(img))
         table.insert(output, image.crop(img, 0, 0, size, size))
         table.insert(output, image.crop(img, w-size, 0, w, size))
         table.insert(output, image.crop(img, 0, h-size, size, h))
         table.insert(output, image.crop(img, w-size, h-size, w, h))
      end

      -- View as mini-batch
      for i, img in ipairs(output) do
         output[i] = img:view(1, img:size(1), img:size(2), img:size(3))
      end

      return input.cat(output, 1)
   end
end

-- Resized with shorter side randomly sampled from [minSize, maxSize] (ResNet-style)
function M.RandomScale(minSize, maxSize)
   return function(input)
      local w, h = input:size(3), input:size(2)

      local targetSz = torch.random(minSize, maxSize)
      local targetW, targetH = targetSz, targetSz
      if w < h then
         targetH = torch.round(h / w * targetW)
      else
         targetW = torch.round(w / h * targetH)
      end

      return image.scale(input, targetW, targetH, 'bicubic')
   end
end

local minArea = 0.7
-- Random crop with size 8%-100% and aspect ratio 3/4 - 4/3 (Inception-style)
function M.RandomSizedCrop(size)
   local scale = M.Scale(size)
   local crop = M.CenterCrop(size)

   return function(input)
      local attempt = 0
      repeat
         local area = input:size(2) * input:size(3)
         local targetArea = torch.uniform(minArea, 1.0) * area

         local aspectRatio = torch.uniform(3/4, 4/3)
         local w = torch.round(math.sqrt(targetArea * aspectRatio))
         local h = torch.round(math.sqrt(targetArea / aspectRatio))

         if torch.uniform() < 0.5 then
            w, h = h, w
         end

         if h <= input:size(2) and w <= input:size(3) then
            local y1 = torch.random(0, input:size(2) - h)
            local x1 = torch.random(0, input:size(3) - w)

            local out = image.crop(input, x1, y1, x1 + w, y1 + h)
            assert(out:size(2) == h and out:size(3) == w, 'wrong crop size')

            return image.scale(out, size, size, 'bicubic')
         end
         attempt = attempt + 1
      until attempt >= 10

      -- fallback
      return crop(scale(input))
   end
end

function M.HorizontalFlip(prob)
   return function(input)
      if torch.uniform() < prob then
         input = image.hflip(input)
      end
      return input
   end
end

function M.Rotation(deg)
   return function(input)
      if deg ~= 0 then
         input = image.rotate(input, (torch.uniform() - 0.5) * deg * math.pi / 180, 'bilinear')
      end
      return input
   end
end

-- Lighting noise (AlexNet-style PCA-based noise)
function M.Lighting(alphastd, eigval, eigvec)
   return function(input)
      if alphastd == 0 then
         return input
      end

      local alpha = torch.Tensor(3):normal(0, alphastd)
      local rgb = eigvec:clone()
         :cmul(alpha:view(1, 3):expand(3, 3))
         :cmul(eigval:view(1, 3):expand(3, 3))
         :sum(2)
         :squeeze()

      input = input:clone()
      for i=1,3 do
         input[i]:add(rgb[i])
      end
      return input
   end
end

local function blend(img1, img2, alpha)
   return img1:mul(alpha):add(1 - alpha, img2)
end

local function grayscale(dst, img)
   dst:resizeAs(img)
   dst[1]:zero()
   dst[1]:add(0.299, img[1]):add(0.587, img[2]):add(0.114, img[3])
   dst[2]:copy(dst[1])
   dst[3]:copy(dst[1])
   return dst
end

function M.Saturation(var)
   local gs

   return function(input)
      gs = gs or input.new()
      grayscale(gs, input)

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      return input
   end
end

function M.Brightness(var)
   local gs

   return function(input)
      gs = gs or input.new()
      gs:resizeAs(input):zero()

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      return input
   end
end

function M.Contrast(var)
   local gs

   return function(input)
      gs = gs or input.new()
      grayscale(gs, input)
      gs:fill(gs[1]:mean())

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      return input
   end
end

function M.RandomOrder(ts)
   return function(input)
      local img = input.img or input
      local order = torch.randperm(#ts)
      for i=1,#ts do
         img = ts[order[i]](img)
      end
      return input
   end
end

function M.ColorJitter(opt)
   local brightness = opt.brightness or 0
   local contrast = opt.contrast or 0
   local saturation = opt.saturation or 0

   local ts = {}
   if brightness ~= 0 then
      table.insert(ts, M.Brightness(brightness))
   end
   if contrast ~= 0 then
      table.insert(ts, M.Contrast(contrast))
   end
   if saturation ~= 0 then
      table.insert(ts, M.Saturation(saturation))
   end

   if #ts == 0 then
      return function(input) return input end
   end

   return M.RandomOrder(ts)
end

local function calc_crops(size_X, size_Y, factor)
   local off = math.floor(factor * size_Y)
   local off5 = math.floor(off * 0.5)
   local w = off * 2 + size_X - 1
   local h = off * 2 + size_Y - 1
   local crop_p4s = {
      -- zoom
      { off, off,   w - off, off,   w - off, h - off,   off, h - off },
      { 0, 0, w, 0, w, h, 0, h},
      { off*2, off*2,   w - off * 2, off * 2,   w - off * 2, h - off * 2,   off * 2, h - off * 2 },
      
      -- perspective crop
      
      -- zoomout
      { off, 0,   w - off, 0,   w - off, h,  off, h},
      { 0, off,   w, off,   w, h - off, 0,   h - off},
      {off, off,   w - off, off,   w, h - off,   0, h - off},
      {0, off,   w, off,   w - off, h - off,   off, h - off},
      {off, 0,   w - off, off,   w - off, h - off,   off, h},
      {off, off,   w - off, 0,    w - off, h, off,   h - off},
      {off, 0,   w - off, off,   w - off, h - off,   off, h},

      -- zoomin
      { off*2, off,   w - off * 2, off,   w - off * 2, h - off,   off * 2, h - off},
      { off, off*2,   w - off, off * 2,    w - off, h - off * 2,    off, h - off * 2},
      {off + off, off + off,   w - off - off, off + off,   w - off, h - off - off,   off, h - off - off},
      {off, off + off,   w - off, off + off,   w - off - off, h - off - off,   off + off, h - off - off},
      {off + off, off,   w - off - off, off + off,   w - off - off, h - off - off,   off + off, h - off},
      {off + off, off + off,   w - off - off, off,   w - off - off, h - off,   off + off, h - off - off},
      {off + off, off,   w - off - off, off + off,   w - off - off, h - off - off,   off + off, h - off},
   }
   return crop_p4s
end

local DA_FACTOR_MIN = 0.1
local DA_FACTOR_MAX = 0.1666

function M.Warp( factor)
   -- jitter for training
  return function(input)
   factor = factor or torch.uniform(DA_FACTOR_MIN, DA_FACTOR_MAX)
   local crop_p4s = calc_crops(input:size(3), input:size(2), factor)
   local p4 = crop_p4s[torch.random(1, #crop_p4s)]
   local off = math.floor(factor * input:size(2))
   input = iproc.zero_padding(input, off)    

   input = iproc.perspective_crop(input,
            p4[1], p4[2],
            p4[3], p4[4],
            p4[5], p4[6],
            p4[7], p4[8],
            input:size(3), input:size(2))

   return input
  end
end

-- change to random color space and add it like a noise
-- can not get it working
function M.ColorSpaceJitter( var)
   local gs
   local colorSpace = {
      image.rgb2lab,
      image.rgb2yuv,
      image.rgb2hsl,
      image.rgb2hsv,
   }

   return function(input)
      local func = colorSpace[torch.random(1, #colorSpace)]
      gs = func(input)
      -- normalize value to range 0-1 of choosen color space
      for i=1,3 do
         gs[i] = (gs[i] - gs[i]:min())/(gs[i]:max() - gs[i]:min())
      end
      -- print ('start')
      -- print (input[1]:min())
      -- print (input[2]:min())
      -- print (input[3]:min())
      local alpha = 1.0 + torch.uniform(0, var)
      blend(input, gs, alpha)
      -- print ('start 2')
      -- print (input[1]:min())
      -- print (input[2]:min())
      -- print (input[3]:min())
      -- image.save(alpha .. "_image.jpg", input)
      return input
   end
end

-- From Baidu: http://arxiv.org/vc/arxiv/papers/1501/1501.02876v1.pdf
-- add +- ~20 values to random channels
function M.ColorCasting( var)
   return function(input)
      local bolCast = torch.Tensor(3):random(0,1) 
      local gs      = input:clone() -- set change as origian image
      for i=1,3 do 
         if bolCast[i]==1 then  
            gs[{i}] = 1.0  -- set maximum value 
         end 
      end
      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      -- image.save(alpha .. "_image.jpg", input)
      return input
   end
end

return M
