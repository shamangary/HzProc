local t = require 'hzproc.online'

local onlineAug, Parent = torch.class('nn.onlineAug', 'nn.Module')

function onlineAug:__init()
   Parent.__init(self)

   self.train = true
   -- The following color jittering params for demo are taken from 
   -- fb.resnet.torch
   self.meanstd = {
     mean = { 0.485, 0.456, 0.406 },
     std = { 0.229, 0.224, 0.225 },
   }
   self.pca = {
     eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
     eigvec = torch.Tensor{
       { -0.5675,  0.7192,  0.4009 },
       { -0.5808, -0.0045, -0.8140 },
       { -0.5836, -0.6948,  0.4203 },
     },
   }
   self.myconverter = self:convert()

end

function onlineAug:convert()
   return t.Compose{
    t.Warp(0.3, 20, 1.2, 1.2),
    --[[
    t.RandomCrop(224, 0),
    t.RandomSizedCrop(55),
    t.ColorJitter({
      brightness = 0.4,
      contrast = 0.4,
      saturation = 0.4,
    }),
    t.Lighting(0.5, self.pca.eigval, self.pca.eigvec),
    t.ColorNormalize(self.meanstd),
    --]]
    t.HorizontalFlip(0.5),
   }
end

function onlineAug:cuda()
end

function onlineAug:updateOutput(input)

   self.output = input.new():resizeAs(input):fill(0)
   if self.train then
      for i=1,input:size(1) do
         -- load the image
         I_input = input[{1,{},{},{}}]         
         I_output = self.myconverter(I_input)
         self.output[{1,{},{},{}}] = I_output:clone()
      end
      
   else
      self.output = input
   end

   
   return self.output
end


