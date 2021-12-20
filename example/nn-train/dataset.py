from math import pow
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DensityToPotentialDataset(Dataset):
    def __init__(self, raw_data, input_channel=4):
        super(DensityToPotentialDataset, self).__init__()
        self.data_np = raw_data.astype(np.float32)
        self.input_channel = input_channel

    def __len__(self):
        return len(self.data_np)

    def __getitem__(self, idx):
        if isinstance(idx, slice): 
            raise ValueError("slice may cause problem in DensityToPotentialDataset.__getitem__()")
        data_slice = np.asarray(self.data_np[slice(idx, idx + 1, None)])
        ndim = int(round(pow(float(data_slice.shape[1] - 1) / float(self.input_channel),  1. / 3)))
        rho = data_slice[:, :-1].reshape(self.input_channel, ndim, ndim, ndim)
        v = data_slice[:, -1]
        rho, v = torch.from_numpy(rho), torch.from_numpy(v)
        return {"rho": rho, "v": v}


class DensityToPotentialDataset_zsym(Dataset):
    def __init__(self, raw_data, input_channel=4):
        super(DensityToPotentialDataset_zsym, self).__init__()
        self.data_np = raw_data
        self.input_channel = input_channel

    def __len__(self):
        return len(self.data_np)

    def __getitem__(self, idx):
        if isinstance(idx, slice): 
            raise ValueError("slice may cause problem in DensityToPotentialDataset.__getitem__()")
        data_slice = np.asarray(self.data_np[slice(idx, idx + 1, None)])
        ndim = int(round(pow(float(data_slice.shape[1] - 1) / float(self.input_channel),  1. / 3)))

        rho_zp = data_slice[:, :-1].reshape(self.input_channel, ndim, ndim, ndim)
        rho_zm = np.empty_like(rho_zp)
        for ic in range(self.input_channel):
            for iz in range(ndim):
                rho_zm[ic, :, :, iz] = rho_zp[ic, :, :, -(iz+1)]
        rho_zm[-1, :, :, :] *= -1
        
        rho = np.array([rho_zp, rho_zm])
        assert(rho.shape == (2, self.input_channel, ndim, ndim, ndim))

        v = data_slice[:, -1]
        rho, v = torch.from_numpy(rho), torch.from_numpy(v)
        return {"rho": rho, "v": v}


def get_train_and_validate_set(options, toDataLoader=True):
    raw_data = np.load(options["data_path"])
    np.random.shuffle(raw_data)

    constrain = options["constrain"]
    if constrain.find("zsym") != -1:
        train_set = DensityToPotentialDataset_zsym(
                raw_data[:options["train_set_size"]], 
                input_channel=options["input_channel"])
        validate_set = DensityToPotentialDataset_zsym(
            raw_data[options["train_set_size"] : options["train_set_size"] + options["validate_set_size"]], 
            input_channel=options["input_channel"])

    else:
        # constrain == "none"
        train_set = DensityToPotentialDataset(
                raw_data[:options["train_set_size"]], 
                input_channel=options["input_channel"])
        validate_set = DensityToPotentialDataset(
            raw_data[options["train_set_size"] : options["train_set_size"] + options["validate_set_size"]], 
            input_channel=options["input_channel"])

    if not toDataLoader: return train_set, validate_set

    train_set_loader = DataLoader(
            train_set, 
            batch_size=int(options["batch_size"]), 
            shuffle=options["shuffle"], 
            num_workers=int(options["num_workers"]))

    validate_set_loader = DataLoader(
            validate_set, 
            batch_size=int(options["batch_size"]), 
            shuffle=options["shuffle"], 
            num_workers=int(options["num_workers"]))

    return train_set_loader, validate_set_loader
    
def get_test_set(options, toDataLoader=True):
    raw_data = np.load(options["data_path"])
    
    constrain = options["constrain"]
    if constrain.find("zsym") != -1:
        test_set = DensityToPotentialDataset_zsym(
                raw_data[:options["test_set_size"]], 
                input_channel=options["input_channel"])
    else:
        test_set = DensityToPotentialDataset(
                raw_data[:options["test_set_size"]], 
                input_channel=options["input_channel"])

    if not toDataLoader: return test_set

    test_set_loader = DataLoader(
            test_set, 
            batch_size=1, 
            shuffle=False, 
            num_workers=int(options["num_workers"]))

    return test_set_loader
    
