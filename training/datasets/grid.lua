require 'image'

local iproc = require './iproc' 


local DA_FACTOR = 0.125
local DA_FACTOR_MIN = 0.1
local DA_FACTOR_MAX = 0.1666

local IMG_SIZE_X = 341
local IMG_SIZE_Y = 256

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

function save_images(x, n, file)
   file = file or "./out.png"
   local input = x:narrow(1, 1, n)
   local view = image.toDisplayTensor({input = input,
               padding = 2,
               nrow = 8,
               symmetric = true})
   image.save(file, view)
end

function test_crop()
   local src = image.load("img_1.jpg") --image.lena():narrow(1, 1, 1)--
   local org_size_Y = src:size(2)
   local org_size_X = src:size(3)
   print (org_size_X .. " " .. org_size_Y)
   local off = DA_FACTOR * org_size_Y
   local crop_p4s = calc_crops( org_size_X, org_size_Y,  DA_FACTOR)
   local imgs = torch.Tensor(#crop_p4s, 3, IMG_SIZE_Y, IMG_SIZE_X )
   src = iproc.zero_padding(src, off)   
   image.save(0 .. "_ff.jpg",  src)
   for i = 1, imgs:size(1) do
      local p4 = crop_p4s[i]
      imgs[i]:copy(
        iproc.perspective_crop(src,
              p4[1], p4[2],
              p4[3], p4[4],
              p4[5], p4[6],
              p4[7], p4[8],
              IMG_SIZE_X, IMG_SIZE_Y )
      )
      image.save(i .. "_ff.jpg",  imgs[i])
 
      
   end
   save_images(imgs, imgs:size(1), "jitter_crop.png")
end