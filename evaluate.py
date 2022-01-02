import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils   

from model import Model
from data import getTrainingTestingData
from utils import AverageMeter, DepthNorm, colorize

def createModel(bestModelPath = None):
  model = Model()
  if bestModelPath != None:
    loadedModelState = torch.load(bestModelPath)
    for name,param in model.named_parameters():
      if param.requires_grad:
        param.data = loadedModelState[name]
  
  model = model.cuda()
  print("Model Created :", bestModelPath)
  return model

def compute_errors(gt, pred):
  bn, an, rn, cn = gt.shape 
  rc = int(rn/10)
  cc = int(cn/10)
  a1 = np.array([])
  a2 = np.array([])
  a3 = np.array([])
  abs_rel = np.array([])
  rmse = np.array([])
  log_10 = np.array([])

  for r in range(9):
    for c in range(9):
      tempGt = gt[:,:,r*rc:(r+1)*rc, c*cc:(c+1)*cc]
      tempPred = pred[:,:,r*rc:(r+1)*rc, c*cc:(c+1)*cc]

      thresh = np.maximum((tempGt / tempPred), (tempPred / tempGt))
      
      a1 = np.append( a1, [(thresh < 1.25   ).mean()])
      a2 = np.append(a2 , [(thresh < 1.25 ** 2).mean()])
      a3 = np.append(a3, [(thresh < 1.25 ** 3).mean()])

      abs_rel = np.append(abs_rel, [np.mean(np.abs(tempGt - tempPred) / tempGt)])

      trmse = (tempGt - tempPred) ** 2
      trmse = np.sqrt(trmse.mean())
      rmse = np.append(rmse, [trmse])
      log_10 = np.append(log_10, [(np.abs(np.log10(tempGt)-np.log10(tempPred))).mean()])

  return np.mean(a1), np.mean(a2), np.mean(a3), np.mean(abs_rel), np.mean(rmse), np.mean(log_10)

def evaluate(model, test_loader):
  model.eval()
  sequential = test_loader
  depth_scores = np.zeros((len(test_loader), 6))
  for j,sample_batched in enumerate(sequential):
    image = torch.autograd.Variable(sample_batched['image'].cuda())
    depth = sample_batched['depth']
    output = DepthNorm(model(image))
    errors = compute_errors(depth.numpy(), output.cpu().detach().numpy())
    temp = np.zeros(6)
    for i,e in enumerate(errors):
      temp[i] = e
    depth_scores[j] = temp
  
  e = depth_scores.mean(axis=0)
  
  print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
  print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0],e[1],e[2],e[3],e[4],e[5]))
  
def main(folderPth ='./saveModels/', batch_size = 5):

  _, test_loader = getTrainingTestingData(batch_size=batch_size)
  files = os.listdir(folderPth)
  for fname in files:
    filePth = folderPth+fname
    model = createModel(filePth)
    evaluate(model, test_loader)
  
if __name__ == '__main__':
    main()