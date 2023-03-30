import torch

from utils import *

from options import get_evaluate_arguments
from config import Config

from data.instances import Instances

from models.utils import build_model

from checker import Evaluate_Checker
from logger import Evaluate_Logger

def main():
    args = get_evaluate_arguments()
    checker = Evaluate_Checker(args.checkpoint_path)
    reset_args(args, checker.state_dict["args"])

    config = Config(args)
    if config.use_cuda:
        torch.cuda.set_device(0)

    set_seed(args.seed, config.use_cuda)

    # preprocess data
    label2idx, char2idx = load_dicts(args.data_dir, is_bert=config.is_bert)
    idx2label, idx2char = get_idx2sth([label2idx, char2idx])
    train_insts, valid_insts, test_insts = Instances.load_data(args.data_dir, args.max_length)

    if args.remove_middle:
        train_insts.remove_middle(idx2label, label2idx)
        valid_insts.remove_middle(idx2label, label2idx)
        test_insts.remove_middle(idx2label, label2idx)

    config.reset_data_table(label2idx, idx2label, char2idx, idx2char)
    print(f"Class number of labels, chars: {config.label_size} {config.num_chars}")
    print(f"Labels: {config.label2idx}")

    if args.use_pretrained_char:
        config.load_char_embeds(load_char_embeds(args.data_dir))
    if args.use_bichar: # load pretrained bichar as placeholders
        config.load_bichar_embeds(load_bichar_embeds(args.data_dir))

    # train and eval model
    train_itr = train_insts.batch_to_tensors(args.max_sentences, args.use_seg, args.use_bichar)
    valid_itr = valid_insts.batch_to_tensors(args.max_sentences, args.use_seg, args.use_bichar)
    test_itr = test_insts.batch_to_tensors(args.max_sentences, args.use_seg, args.use_bichar)

    model = build_model(config)
    checker.load_model(model)

    if config.use_cuda:
        model = model.cuda()

    checker.delete_model_state()


    # Logger: responsible for logging and printing
    logger = Evaluate_Logger(config)

    # train model
    epoch = checker.start_epoch()
    update = checker.num_updates()

    alphas = args.alphas.split(" ")

    for alpha_s in alphas:
        alpha = float(alpha_s)
        model.set_alpha(alpha)

        def _evaluate(data_itr, model):
            model.eval()

            log_outputs = []
            for sample in data_itr:
                sample = move_to_cuda(sample)
                loss, sample_size, log_output = model.evaluate(sample, idx2label)
                log_outputs.append(log_output)
            log_output = model.aggregate_log_outputs(log_outputs)
            return log_output

        # train_log_output = _evaluate(train_itr, model)
        valid_log_output = _evaluate(valid_itr, model)
        test_log_output = _evaluate(test_itr, model)

        logger.print_result(epoch, update, valid_log_output, test_log_output, alpha)

    logger.save_log()

if __name__ == "__main__":
    main()

















