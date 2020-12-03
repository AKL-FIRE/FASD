import numpy as np
import random

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms

torch.manual_seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(data_path, label):
	data = np.load(data_path, allow_pickle=True)
	x = []
	for person in data:
		person_data = []
		tmp = person[0]
		if len(tmp) != len(data[0][0]):
			continue
		for t in tmp:
			person_data.append([t[0]/1000, t[1]/1000])
		x.append([torch.Tensor(person_data), label])
	return x


class simpleLSTM(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes, num_layers=1):
		super(simpleLSTM, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
		self.fc = nn.Linear(hidden_size, num_classes)

	def forward(self, x):
		# x shape (batch, time_step, input_size)
		# out shape (batch, time_step, output_size)
		# h_n shape (n_layers, batch, hidden_size)
		# h_c shape (n_layers, batch, hidden_size)
		h0 = torch.rand(self.num_layers, x.size(0), self.hidden_size).to(device)
		c0 = torch.rand(self.num_layers, x.size(0), self.hidden_size).to(device)

		out, (h_n, h_c) = self.lstm(x, (h0, c0))

		out = self.fc(out[:, -1, :])

		return out

def train(train_data, model, optimizer, criterion, batch_size, n_iters, n_print):
	for it in range(n_iters):

		inputs = torch.zeros(batch_size, train_data[0][0].shape[0], 2)
		labels = torch.tensor([0 for i in range(batch_size)])
		for i in range(batch_size):
			_x,_label = random.choice(train_data)
			inputs[i] = _x
			labels[i] = _label
		inputs = inputs.to(device)
		labels = labels.to(device)

		# forward pass
		outputs = model(inputs)
		loss = criterion(outputs, labels)

		# backward and optimize
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
			
		if it % n_print == 0:
			print('Iter [{}/{}], Loss: {:.4f}'.format(it+1, n_iters, loss.item()))

def test(test_data):
	model.eval()
	with torch.no_grad():

		inputs = torch.zeros(batch_size, train_data[0][0].shape[0], 2)
		labels = torch.zeros(batch_size)
		for i, (_x, _label) in enumerate(test_data):
			inputs[i] = _x
			labels[i] = _label
		inputs = inputs.to(device)
		labels = labels.to(device)

		outputs = model(inputs)
		_, predicted = torch.max(outputs, 1)
		total = labels.size(0)
		correct = (predicted == labels).sum().item()
		print('Test Accuracy: {} %'.format(100 * correct / total))

if __name__ == '__main__':
	control_data = load_data('./data/control_rawdata.npy', 0)
	patient_data = load_data('./data/fasd_rawdata.npy', 1)
	n_iters = 1000
	n_print = 50        
	batch_size = 10
	time_step = 2216      
	input_size = 2   
	num_classes = 2  
	hidden_size = 10
	num_layers = 1
	lr = 0.01 

	test_data = []
	test_data.extend(control_data[-5:])
	test_data.extend(patient_data[-5:])
	train_data = []
	train_data.extend(control_data[:-5])
	train_data.extend(patient_data[:-5])


	model = simpleLSTM(input_size, hidden_size, num_classes).to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr)

	train(train_data, model, optimizer, criterion, batch_size, n_iters, n_print)

	test(test_data)
