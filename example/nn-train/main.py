from __future__ import print_function
import os, sys
import torch.optim as optim

from const_list import *
from Config import get_options
from dataset import *
from func import *
from log_to_file import Logger

def main():
    options = get_options(sys.argv[1])

    logger = Logger(options["log_path"], to_stdout=options["verbose"])
    logger.log("========Task Start========")

    train_set_loader, validate_set_loader = \
            get_train_and_validate_set(options)

    model = get_model(options["model"])()
    if "restart" in options.keys():
        load_model(model, options["restart"])
    if options["enable_cuda"]:
        model.cuda()

    if options["loss_function"] == "MSELoss_zsym":
        if "zsym_coef" in options.keys():
            loss_func = get_list_item(LOSS_FUNC_LIST, options["loss_function"])(float(options["zsym_coef"]))
        else:
            loss_func = get_list_item(LOSS_FUNC_LIST, options["loss_function"])()
    else:
        loss_func = get_list_item(LOSS_FUNC_LIST, options["loss_function"])()
    loss_func.size_average = True

    optimiser = get_list_item(OPTIM_LIST, options["optimiser"])(model.parameters(), lr=options["learning_rate"])

    # start logger
    logger.log(str(model), "main")
    logger.log("Max iteration: %d" % (options["max_epoch"]), "main")
    logger.log("Learning rate: %e" % (options["learning_rate"]), "main")
    logger.log("Loss function", "main")
    logger.log(str(loss_func), "main")
    logger.log("Optimiser", "main")
    logger.log(str(optimiser), "main")
    logger.log("Model saved to %s" % (options["model_save_path"]), "main")
    os.system("mkdir -p %s" % (options["model_save_path"][:options["model_save_path"].rfind('/')]))

    n_restart = int(options["n_restart"]) if "n_restart" in options.keys() else 20

    # Train!!
    logger.log("Train start", "main")
    
    train_x = np.array(range(options["max_epoch"]))
    train_y = np.zeros(len(train_x), dtype=np.float32)
    validate_x = np.array(range(options["max_epoch"]))
    validate_y = np.zeros(len(validate_x), dtype=np.float32)

    for epoch in range(options["max_epoch"]):
        loss_on_train = train(epoch, train_set_loader, model, loss_func, optimiser, logger, cuda=options["enable_cuda"])
        loss_on_validate = validate(epoch, validate_set_loader, model, loss_func, logger, cuda=options["enable_cuda"])

        if epoch % (n_restart) == n_restart-1:
            save_model(model, options["model_save_path"] + ".restart%d" % (epoch+1))
        train_y[epoch] = loss_on_train
        validate_y[epoch] = loss_on_validate

    save_model(model, options["model_save_path"])
    logger.log("Model saved.", "main")

    logger.log("========Task Finish========")

if __name__ == "__main__":
    main()

