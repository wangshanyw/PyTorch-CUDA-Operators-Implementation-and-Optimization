import torch
import time

matrix = torch.rand(2000,2000).to(torch.float).cuda()

start = time.time()
torch.cuda.synchronize()
softmax = torch.nn.Softmax(dim=0)

for i in range(100):
    res = softmax(matrix)

torch.cuda.synchronize()
end = time.time()

print("Softmax elapsed time: {:.4f}".format(end - start))
