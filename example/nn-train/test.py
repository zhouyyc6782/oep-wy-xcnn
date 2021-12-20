from __future__ import print_function
import datetime 
import os, sys
import numpy
import torch.optim as optim

from const_list import *
from Config import get_options
from dataset import *
from func import *
from log_to_file import Logger

def main():
    # options = get_options(sys.argv[1])
    for opts in sys.argv[1:]:
        # print(opts)
        options = get_options(opts)

        os.system("mkdir -p %s" % (options["log_path"][:options["log_path"].rfind('/')]))
        logger = Logger(options["log_path"], to_stdout=options["verbose"])
        logger.log("========Task Start========")

        test_set_loader = get_test_set(options)

        model = get_model(options["model"])()

        if options["loss_function"] == "MSELoss_zsym":
            if options.has_key("zsym_coef"):
                loss_func = get_list_item(LOSS_FUNC_LIST, options["loss_function"])(float(options["zsym_coef"]))
            else:
                loss_func = get_list_item(LOSS_FUNC_LIST, options["loss_function"])()
        else:
            loss_func = get_list_item(LOSS_FUNC_LIST, options["loss_function"])()
        loss_func.size_average = True

        load_model(model, options["restart"])
        if options["enable_cuda"]:
            model.cuda()

        # start logger
        logger.log(str(model), "main")
        logger.log("Loss function", "main")
        logger.log(str(loss_func), "main")
        logger.log("Model loaded from %s" % (options["restart"]), "main")

        # Test!!
        logger.log("Test start", "main")
        
        loss_on_test, test_output, test_target = \
                test(test_set_loader, model, loss_func, logger, cuda=options["enable_cuda"])
        
        os.system("mkdir -p %s" % (options["output_path"]))
        save_ndarray(test_output, 
                "%s/%s_output" % 
                (options["output_path"], 
                    options["restart"].split("/")[-1]
                ))
        save_ndarray(test_target, 
                "%s/%s_target" % 
                (options["output_path"], 
                    options["restart"].split("/")[-1]
                ))


        logger.log("========Task Finish========")

if __name__ == "__main__":
    main()

