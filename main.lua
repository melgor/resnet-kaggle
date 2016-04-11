--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'models/ResidualDrop'
local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)
cutorch.setDevice(1)
-- 
-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)
local optimState = checkpoint and torch.load('logs_resnet50_256/' .. checkpoint.optimFile) or nil

-- Create model
local model, criterion = models.setup(opt, checkpoint)

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)
-- 
if opt.testOnly then
   local top1Err, top5Err = trainer:test(0, valLoader)
   print(string.format(' * Results top1: %6.3f  top5: %6.3f', top1Err, top5Err))
   return
end

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestTop1 = math.huge
local bestTop5 = math.huge
local bestLoss = math.huge
for epoch = startEpoch, opt.nEpochs do
   -- Train for a single epoch
   local trainTop1, trainTop5, trainLoss = trainer:train(epoch, trainLoader)

   -- Run model on validation set
   local testTop1, testTop5, testLoss = trainer:test(epoch, valLoader)

   local bestModel = false
   if testLoss < bestLoss then
      bestModel = true
      bestTop1 = testTop1
      bestTop5 = testTop5
      bestLoss = testLoss
      print(' * Best model ', testTop1, testTop5)
   end
   --save Logs
   checkpoints.saveJSON(epoch, trainer.optimState, opt, {
      train_err = trainTop1,
      train_err5 = trainTop5,
      train_accuracy = 100 - trainTop1,
      train_loss = trainLoss,
      test_err = testTop1,
      test_err5 = testTop5,
      test_accuracy = 100 - testTop1,
      test_loss = testLoss,
      best_err = bestTop1,
      best_err5 = bestTop5,
      best_loss = bestLoss,
      best_epochNumber = epoch,
      best_trainLoss = trainLoss,
      best_accuracy = 100 - bestTop1,
      finished = false,
   })
   checkpoints.save(epoch, model, trainer.optimState, bestModel, opt)
end

print(string.format(' * Finished top1: %6.3f  top5: %6.3f', bestTop1, bestTop5))
