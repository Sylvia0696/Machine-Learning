# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import torch.optim as optim
# import torch

# # word
# word_dict = {}
# with open('../data/vocabs.word') as wf:
#     wlines = wf.readlines()

# for wline in wlines:
#     wpart = wline.split()
#     word_dict[wpart[0]] = int(wpart[1])


# # pos
# pos_dict = {}
# with open('../data/vocabs.pos') as pf:
#     plines = pf.readlines()

# for pline in plines:
#     ppart = pline.split()
#     pos_dict[ppart[0]] = int(ppart[1])


# # labels
# labels_dict = {}
# with open('../data/vocabs.labels') as lf:
#     llines = lf.readlines()

# for lline in llines:
#     lpart = lline.split()
#     labels_dict[lpart[0]] = int(lpart[1])


# # actions
# actions_dict = {}
# with open('../data/vocabs.actions') as af:
#     alines = af.readlines()

# for aline in alines:
#     apart = aline.split()
#     actions_dict[apart[0]] = int(apart[1])


# # train.data to a matrix with nums
# train_x = []
# train_y = []
# with open('../data/train.data') as trf:
#     trlines = trf.readlines()

# for trline in trlines:
#     trpart = trline.split()
#     line_list = []
#     for i in range(0, 20):
#         if trpart[i] in word_dict.keys():
#             line_list.append(word_dict[trpart[i]])
#         else:
#             line_list.append(word_dict['<unk>'])
#     for i in range(20, 40):
#         line_list.append(pos_dict[trpart[i]])
#     for i in range(40,52):
#         line_list.append(labels_dict[trpart[i]])
#     train_x.append(line_list)
#     train_y.append(actions_dict[trpart[52]])

# train_x = np.array([l for l in train_x])
# train_y = np.array(train_y)

# # print(type(train_x))
# # print(type(train_y))
# # print(train_x.shape)
# # print(train_y.shape)
# # <class 'numpy.ndarray'>
# # <class 'numpy.ndarray'>
# # (143758, 52)
# # (143758,)



# class Net(nn.Module):
#     def __init__(self, word_size, word_em, pos_size, pos_em, label_size, label_em, action_size):
#         super(Net, self).__init__()
#         self.word_embeddings = nn.Embedding(word_size, word_em)
#         self.pos_embeddings = nn.Embedding(pos_size, pos_em)
#         self.label_embeddings = nn.Embedding(label_size, label_em)
#         self.linear1 = nn.Linear(20*word_em + 20*pos_em + 12*label_em, 200)
#         self.linear2 = nn.Linear(200, 200)
#         self.linear3 = nn.Linear(200, action_size)

#     def forward(self, inputs):
#         word_embeds = self.word_embeddings(inputs[0:20]).view((1, -1))
#         pos_embeds = self.pos_embeddings(inputs[20:40]).view((1, -1))
#         label_embeds = self.label_embeddings(inputs[40:52]).view((1, -1))
#         embeds = torch.cat((word_embeds,pos_embeds,label_embeds),1)
#         out = F.relu(self.linear1(embeds))
#         out = F.relu(self.linear2(out))
#         out = self.linear3(out)
#         log_probs = F.log_softmax(out, dim=1)
#         return log_probs


# net = Net(4807,64,45,32,46,32,93)


# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# trainloader=torch.utils.data.DataLoader(train_x, batch_size=1000, shuffle=True, num_workers=8)


# for epoch in range(7):  # loop over the dataset multiple times

#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         inputs, labels = data    

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize

#         outputs = net(inputs)
#         loss = criterion(outputs, torch.tensor([data_y], dtype=torch.long))
#         loss.backward()
#         optimizer.step()

#         # print statistics
#         running_loss += loss.item()
#         print(loss.item())
# print('Finished Training')



from network import *
from net_properties import *

net_properties = NetProperties(64, 32, 32, 200, 200, 1000)
network = Network(net_properties)

network.load('network1.model')