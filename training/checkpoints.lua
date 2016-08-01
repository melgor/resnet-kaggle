--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
json = require('json') 
local checkpoint = {}

function checkpoint.latest(opt)
   if opt.resume == 'none' then
      return nil
   end

   local latestPath = paths.concat(opt.resume, 'latest.t7')
   if not paths.filep(latestPath) then
      return nil
   end

   print('=> Loading checkpoint ' .. latestPath)
   local latest = torch.load(latestPath)
   local optimState = torch.load(paths.concat(opt.resume, latest.optimFile))
   return latest, optimState
end

function checkpoint.save(epoch, model, optimState, bestModel, opt)
   -- Don't save the DataParallelTable for easier loading on other machines
   if torch.type(model) == 'nn.DataParallelTable' then
      model = model:get(1)
   end

   local modelFile = opt.logs .. '/model_' .. epoch .. '.t7'
   local optimFile = opt.logs .. '/optimState_' .. epoch .. '.t7'
   
   -- if epoch % 5 == 0 then
   --  torch.save(modelFile, model)
   --  torch.save(optimFile, optimState)
   --  torch.save(opt.logs .. '/latest.t7', {
   --      epoch = epoch,
   --      modelFile = modelFile,
   --      optimFile = optimFile,
   --  })
   -- end

   if bestModel then
      torch.save(opt.logs ..'/model_best.t7', model)
   end
end


function checkpoint.saveJSON( epoch, optimState, opt,  data)
   local name_file = opt.logs .. '/log_' .. epoch .. '.json'
   data['optimState']  = optimState
   data['opt']  = opt
   json.save(name_file, data)
end

return checkpoint
