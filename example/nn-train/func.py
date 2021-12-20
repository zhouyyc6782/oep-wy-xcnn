import time

import numpy as np
import torch
from torch.autograd import Variable

from const_list import MODEL_LIST

def get_model(model_name):
    try:
        model = MODEL_LIST[model_name.lower()]
    except KeyError:
        print("No model named %s" % (model_name.lower()))
        exit(1)
    else:
        pass
    return model

def get_list_item(LIST, key):
    try:
        result = LIST[key.lower()]
    except KeyError:
        result = LIST["default"]
    else:
        pass
    return result

def train(epoch, train_set_loader, model, loss_func, optimiser, logger=None, panel=None, cuda=False):
    time_start = time.time()
    batch_size = train_set_loader.batch_size
    running_loss = 0.
    for batch_idx, data in enumerate(train_set_loader):
        inputs, targets = data["rho"], data["v"]
        if cuda: 
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        
        optimiser.zero_grad()

        outputs = model(inputs)
        loss = loss_func(outputs, targets)
        loss.backward()
        running_loss += loss.data.item()

        optimiser.step()
        
        if logger is not None and batch_idx % 50 == 49:
            logger.log("train batch %5d: [%5d/%5d]\tloss: %.8e" % 
                    (batch_idx + 1, batch_idx * batch_size, len(train_set_loader.dataset), loss.data.item()), "train")

    if logger is not None:
        logger.log("Epoch %5d: average loss on train: %.8e" % 
                (epoch, running_loss * batch_size / float(len(train_set_loader.dataset))), "train")
        logger.log("elapse time: %lf" % (time.time() - time_start), "train")

    return running_loss

def validate(epoch, validate_set_loader, model, loss_func, logger=None, cuda=False):
    time_start = time.time()
    batch_size = validate_set_loader.batch_size
    running_loss = 0.
    for batch_idx, data in enumerate(validate_set_loader):
        inputs, targets = data["rho"], data["v"]
        if cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)

        outputs = model(inputs)
        loss = loss_func(outputs, targets)
        running_loss += loss.data.item()

        if logger is not None and batch_idx % 50 == 49:
            logger.log("validate batch %5d: [%5d/%5d]\tloss: %.8e" % 
                    (batch_idx + 1, batch_idx * batch_size, len(validate_set_loader.dataset), loss.data.item()), "validate")

    if logger is not None:
        logger.log("Epoch %5d: average loss on validate: %.8e" % 
                (epoch, running_loss * batch_size / float(len(validate_set_loader.dataset))), "validate")
        logger.log("elapse time: %lf" % (time.time() - time_start), "validate")

    return running_loss

def test(test_set_loader, model, loss_func, logger=None, cuda=False):
    time_start = time.time()
    batch_size = test_set_loader.batch_size # should be 1
    running_loss = np.zeros([len(test_set_loader.dataset)])
    running_output = np.empty(len(test_set_loader.dataset))
    running_target = np.empty(len(test_set_loader.dataset))
    for batch_idx, data in enumerate(test_set_loader):
        inputs, targets = data["rho"], data["v"]
        if cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)

        outputs = model(inputs)
        loss = loss_func(outputs, targets)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        running_loss[batch_idx] = loss.data.item()
        running_output[batch_idx] = outputs.data[0][0]
        running_target[batch_idx] = targets.data[0][0]

        if logger is not None:
            logger.log("test batch %5d: [%5d/%5d]\tloss: %.8e" % 
                    (batch_idx + 1, batch_idx * batch_size, len(test_set_loader.dataset), loss.data.item()), "test")

    if logger is not None:
        logger.log("Average loss on test: %.8e" % 
                (np.mean(running_loss)), "test")
        logger.log("elapse time: %lf" % (time.time() - time_start), "test")

    return running_loss, running_output, running_target

# IO
def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

def load_model(model, load_path):
    model.load_state_dict(torch.load(load_path))

def save_ndarray(target, save_path):
    np.save(save_path, target)

