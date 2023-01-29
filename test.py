import torch
import torch.nn as nn

from time import process_time_ns

torch.backends.cudnn.enabled = False

test_conv1 = nn.Conv2d(in_channels=5, out_channels=27, kernel_size=3, stride=1, padding=1)
test_conv1_cuda = nn.Conv2d(in_channels=5, out_channels=27, kernel_size=3, stride=1, padding=1).cuda()
test_input1 = torch.rand(1024, 5, 90, 90)
test_input1_cuda = test_input1.cuda()
out1 = test_conv1(test_input1)
t0 = process_time_ns()
out1_cuda = test_conv1_cuda(test_input1_cuda)
t1 = process_time_ns()
out1_cuda = out1_cuda.cpu()

test_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1)
test_conv2_cuda = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1).cuda()
test_input2 = torch.rand(1024, 16, 192, 128)
test_input2_cuda = torch.rand(1024, 16, 192, 128).cuda()
out2 = test_conv2(test_input2)
t2 = process_time_ns()
out2_cuda = test_conv2_cuda(test_input2_cuda)
t3 = process_time_ns()
out2_cuda = out2_cuda.cpu()

diff1 = torch.mean(out1_cuda - out1).item()
diff2 = torch.mean(out2_cuda - out2).item()

print("================================================")
print("Test1 diff: %s" % (str(diff1)))
print("Test2 diff: %s" % (str(diff2)))
print("Test1 running time: %s ns" % (str(t1 - t0)))
print("Test2 running time: %s ns" % (str(t3 - t2)))
print("================================================")
