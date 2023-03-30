import os
import time
import logging
import numpy as np
import pickle as pkl

from models.utils import build_model, evaluate_each_entity_type, evaluate_edit_distance_between_preds_and_golds, \
    write_wrong_result

logger = logging.getLogger(__name__)

class Train_Logger:

    def __init__(self, cfg):
        self.start_t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.model_name = cfg.model_name

        self.epoch_output = {
            "epoch_start_time": time.time(),
            "loss": 0.0,
            "loss_span": 0.0,
            "loss_node": 0.0,
            "sample_size_span": 0.0,
            "sample_size_node": 0.0,
        }

        self.update_output = {
            "update_start_time": time.time(),
        }

        self.save_log_file = cfg.save_log_file
        self.save_best_result_file = cfg.save_best_result_file
        self.epoch_save_log_list = []
        self.update_save_log_list = []
        self.best_epoch = 0
        self.best_update = 0

        self.t_steps = cfg.t_steps

    def reset_before_epoch(self, total_updates=0):
        self.epoch_output["epoch_start_time"] = time.time()
        self.epoch_output["loss"] = 0.0
        self.epoch_output["loss_span"] = 0.0
        self.epoch_output["loss_node"] = 0.0
        self.epoch_output["sample_size_span"] = 0.0
        self.epoch_output["sample_size_node"] = 0.0

        self.reset_before_update()

    def reset_before_update(self):
        self.update_output["update_start_time"] = time.time()

    def add_log_output(self, log_output):

        self.epoch_output["loss"] += log_output.get("loss", 0)
        self.epoch_output["loss_span"] += log_output.get("loss_span", 0)
        self.epoch_output["loss_node"] += log_output.get("loss_node", 0)
        self.epoch_output["sample_size_span"] += log_output.get("sample_size_span", 0)
        self.epoch_output["sample_size_node"] += log_output.get("sample_size_node", 0)

    def print_at_epoch(self, epoch, update, lr, total_norm, valid_log_output, test_log_output):

        train_loss = self.epoch_output.get("loss", 0)
        train_loss_span = self.epoch_output.get("loss_span", 0)
        train_loss_node = self.epoch_output.get("loss_node", 0)

        valid_loss = valid_log_output.get("loss", 0)
        valid_loss_span = valid_log_output.get("loss_span", 0)
        valid_loss_node = valid_log_output.get("loss_node", 0)

        test_loss = test_log_output.get("loss", 0)
        test_loss_span = test_log_output.get("loss_span", 0)
        test_loss_node = test_log_output.get("loss_node", 0)

        valid_p = valid_log_output.get("p", 0)
        valid_r = valid_log_output.get("r", 0)
        valid_f = valid_log_output.get("f", 0)

        test_p = test_log_output.get("p", 0)
        test_r = test_log_output.get("r", 0)
        test_f = test_log_output.get("f", 0)

        logger.info(
            "Epoch={} ".format(epoch)+
            "upd={}/{} ".format(update, self.t_steps)+
            "\n"+
            "[train] loss={:.4f} ".format(train_loss)+
            "loss_span={:.4f} ".format(train_loss_span)+
            "loss_node={:.4f} ".format(train_loss_node)+
            "\n" +
            "[valid] loss={:.4f} ".format(valid_loss) +
            "loss_span={:.4f} ".format(valid_loss_span) +
            "loss_node={:.4f} ".format(valid_loss_node) +
            "p={:.2f} ".format(valid_p) +
            "r={:.2f} ".format(valid_r) +
            "f={:.2f} ".format(valid_f) +
            "\n" +
            "[test]  loss={:.4f} ".format(test_loss) +
            "loss_span={:.4f} ".format(test_loss_span) +
            "loss_node={:.4f} ".format(test_loss_node) +
            "p={:.2f} ".format(test_p)+
            "r={:.2f} ".format(test_r)+
            "f={:.2f} ".format(test_f)+
            "\n"+
            "lr={:.2e} ".format(lr)+
            "gn={:.2f} ".format(total_norm)+
            "time={:.0f}s. ".format(time.time() - self.epoch_output["epoch_start_time"])+
            "\n"
        )

        log = {
            "epoch": epoch,
            "update": update,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "test_loss": test_loss,
            "valid_p": valid_p,
            "valid_r": valid_r,
            "valid_f": valid_f,
            "test_p": test_p,
            "test_r": test_r,
            "test_f": test_f,
            "lr": lr,
            "grad_norm": total_norm,
            "time={:.0f}s.": time.time() - self.epoch_output["epoch_start_time"]
        }
        self.add_epoch_save_log(log)

    def print_at_update(self, epoch, update, lr, total_norm, valid_log_output, test_log_output):

        valid_loss = valid_log_output.get("loss", 0)
        valid_loss_span = valid_log_output.get("loss_span", 0)
        valid_loss_node = valid_log_output.get("loss_node", 0)

        test_loss = test_log_output.get("loss", 0)
        test_loss_span = test_log_output.get("loss_span", 0)
        test_loss_node = test_log_output.get("loss_node", 0)

        valid_p = valid_log_output.get("p", 0)
        valid_r = valid_log_output.get("r", 0)
        valid_f = valid_log_output.get("f", 0)

        test_p = test_log_output.get("p", 0)
        test_r = test_log_output.get("r", 0)
        test_f = test_log_output.get("f", 0)

        logger.info(
            "- "+
            "epoch={} ".format(epoch)+
            "upd={}/{} ".format(update, self.t_steps)+
            "\n"+
            "[valid] loss={:.4f} ".format(valid_loss)+
            "loss_span={:.4f} ".format(valid_loss_span)+
            "loss_node={:.4f} ".format(valid_loss_node)+
            "p={:.2f} ".format(valid_p) +
            "r={:.2f} ".format(valid_r) +
            "f={:.2f} ".format(valid_f) +
            "\n" +
            "[test]  loss={:.4f} ".format(test_loss)+
            "loss_span={:.4f} ".format(test_loss_span)+
            "loss_node={:.4f} ".format(test_loss_node)+
            "p={:.2f} ".format(test_p)+
            "r={:.2f} ".format(test_r)+
            "f={:.2f} ".format(test_f)+
            "\n"+
            "lr={:.2e} ".format(lr)+
            "gn={:.2f} ".format(total_norm)+
            "time={:.0f}s. ".format(time.time() - self.update_output["update_start_time"])+
            "\n"
        )

        # reset update_start_time
        self.update_output["update_start_time"] = time.time()

        log = {
            "epoch": epoch,
            "update": update,
            "valid_loss": valid_loss,
            "test_loss": test_loss,
            "valid_p": valid_p,
            "valid_r": valid_r,
            "valid_f": valid_f,
            "test_p": test_p,
            "test_r": test_r,
            "test_f": test_f,
            "lr": lr,
            "grad_norm": total_norm,
            "time={:.0f}s.": time.time() - self.epoch_output["epoch_start_time"]
        }
        self.add_update_save_log(log)

    def get_perplexity(self, loss):
        return np.exp(loss)

    def add_epoch_save_log(self, log):
        self.epoch_save_log_list.append(log)

    def add_update_save_log(self, log):
        self.update_save_log_list.append(log)

    def set_best_epoch_update(self, epoch, update):
        self.best_epoch = epoch
        self.best_update = update

    def save_log(self):
        assert self.save_log_file[-1] != "/"
        save_log_dir = self.save_log_file[:self.save_log_file.rfind("/")]
        if not os.path.exists(save_log_dir):
            os.makedirs(save_log_dir)

        f = open(self.save_log_file, "wb")
        pkl.dump([self.best_epoch, self.best_update, self.epoch_save_log_list, self.update_save_log_list], f)
        f.close()

    def print_and_save_best_result(self, cfg, best_val, best_test_by_val, best_test):
        assert self.save_best_result_file[-1] != "/"
        save_best_dir = self.save_best_result_file[:self.save_best_result_file.rfind("/")]
        if not os.path.exists(save_best_dir):
            os.makedirs(save_best_dir)

        def round_scores(s):
            s["p"] = round(s["p"], 2)
            s["r"] = round(s["r"], 2)
            s["f"] = round(s["f"], 2)
            str_s = f"p={s['p']} r={s['r']} f={s['f']}"
            return str_s

        best_result_str = f"Best valid scores: {round_scores(best_val)} \n" \
                          f"Best test scores by valid: {round_scores(best_test_by_val)} \n" \
                          f"Best test scores: {round_scores(best_test)} \n"

        logger.info("\n" + best_result_str)

        f = open(self.save_best_result_file, "a")
        f.write(str(self.start_t) + "\n")
        f.write(str(cfg) + "\n")
        f.write(best_result_str + "\n")
        f.close()


class Evaluate_Logger:

    def __init__(self, cfg):
        self.model_name = cfg.model_name
        self.ebpem_gamma = cfg.ebpem_gamma

        self.eval_each_entity_type = cfg.eval_each_entity_type
        self.entity_types = cfg.entity_types

        self.save_log_file = cfg.save_log_file
        self.save_log_list = []

    def print_result(self, epoch, update, valid_log_output, test_log_output, alpha):
        epison = 1e-13

        ebpem_gamma = self.ebpem_gamma

        valid_backbone_loss = valid_log_output.get("backbone_loss", 0) / (valid_log_output.get("num_tokens", 0) + epison)
        valid_ebpem_loss = valid_log_output.get("ebpem_loss", 0) / (2 * valid_log_output.get("num_be", 0) + epison)
        valid_orig_be_loss = valid_log_output.get("orig_be_loss", 0) / (2 * valid_log_output.get("num_be", 0) + epison)
        valid_loss = valid_backbone_loss + ebpem_gamma * valid_ebpem_loss

        test_backbone_loss = test_log_output.get("backbone_loss", 0) / (test_log_output.get("num_tokens", 0) + epison)
        test_ebpem_loss = test_log_output.get("ebpem_loss", 0) / (2 * test_log_output.get("num_be", 0) + epison)
        test_orig_be_loss = test_log_output.get("orig_be_loss", 0) / (2 * test_log_output.get("num_be", 0) + epison)
        test_loss = test_backbone_loss + ebpem_gamma * test_ebpem_loss

        valid_p = valid_log_output.get("p", 0)
        valid_r = valid_log_output.get("r", 0)
        valid_f = valid_log_output.get("f", 0)

        test_p = test_log_output.get("p", 0)
        test_r = test_log_output.get("r", 0)
        test_f = test_log_output.get("f", 0)

        valid_orig_p = valid_log_output.get("orig_p", 0)
        valid_orig_r = valid_log_output.get("orig_r", 0)
        valid_orig_f = valid_log_output.get("orig_f", 0)

        test_orig_p = test_log_output.get("orig_p", 0)
        test_orig_r = test_log_output.get("orig_r", 0)
        test_orig_f = test_log_output.get("orig_f", 0)

        logger.info(
            "Epoch={}".format(epoch),
            "upd={}".format(update),
            "alpha={:.2f}".format(alpha),
            "[     Valid]",
            "l={:.4f}".format(valid_loss),
            "bl={:.4f}".format(valid_backbone_loss),
            "el={:.4f}".format(valid_ebpem_loss),
            "obel={:.2f}".format(valid_orig_be_loss),
            "[Test]",
            "l={:.2f}".format(test_loss),
            "bl={:.2f}".format(test_backbone_loss),
            "el={:.2f}".format(test_ebpem_loss),
            "obel={:.2f}".format(test_orig_be_loss),
            "\n",
            " " * 20,
            "[     Valid]",
            "p={:.2f}".format(valid_p),
            "r={:.2f}".format(valid_r),
            "f={:.2f}".format(valid_f),
            "[     Test]",
            "p={:.2f}".format(test_p),
            "r={:.2f}".format(test_r),
            "f={:.2f}".format(test_f),
            "\n",
            " " * 20,
            "[Orig Valid]",
            "p={:.2f}".format(valid_orig_p),
            "r={:.2f}".format(valid_orig_r),
            "f={:.2f}".format(valid_orig_f),
            "[Orig Test]",
            "p={:.2f}".format(test_orig_p),
            "r={:.2f}".format(test_orig_r),
            "f={:.2f}".format(test_orig_f),
        )


        # write result
        self.write_wrong_result = True
        data_name = "ontonote4"
        orig_data_file = f"/newNAS/Workspaces/NLPGroup/juncw/summer_dataset/reduce_emb_dataset/{data_name}/dev.txt.clip"
        output_data_file = f"/newNAS/Workspaces/NLPGroup/juncw/summer_dataset/reduce_emb_dataset/__ckpts/{data_name}/tener_crf/wrong_result.txt"
        if self.write_wrong_result:
            write_wrong_result(
                valid_log_output["pred_labels"],
                valid_log_output["gold_labels"],
                orig_data_file,
                output_data_file
            )
            exit(0)


        # evaluate edit distance of preds and golds
        self.eval_edit_dist = True
        self.eval_edit_dist_max_range = 5
        if self.eval_edit_dist:
            dist, ntoken = evaluate_edit_distance_between_preds_and_golds(
                valid_log_output["pred_labels"],
                valid_log_output["gold_labels"],
                max_range=self.eval_edit_dist_max_range
            )
            logger.info("eval valid --- ", "dist:", dist, "ntoken:", ntoken, "dist/token:", round(dist/ntoken, 2))
            dist, ntoken = evaluate_edit_distance_between_preds_and_golds(
                test_log_output["pred_labels"],
                test_log_output["gold_labels"],
                max_range=self.eval_edit_dist_max_range
            )
            logger.info("eval test --- ", "dist:", dist, "ntoken:", ntoken, "dist/token:", round(dist / ntoken, 2))
            dist, ntoken = evaluate_edit_distance_between_preds_and_golds(
                valid_log_output["orig_pred_labels"],
                valid_log_output["gold_labels"],
                max_range=self.eval_edit_dist_max_range
            )
            logger.info("eval orig valid --- ", "dist:", dist, "ntoken:", ntoken, "dist/token:", round(dist / ntoken, 2))
            dist, ntoken = evaluate_edit_distance_between_preds_and_golds(
                test_log_output["orig_pred_labels"],
                test_log_output["gold_labels"],
                max_range=self.eval_edit_dist_max_range
            )
            logger.info("eval orig test --- ", "dist:", dist, "ntoken:", ntoken, "dist/token:", round(dist / ntoken, 2))


        if self.eval_each_entity_type:
            logger.info("eval valid")
            valid_res_each_type = evaluate_each_entity_type(
                valid_log_output["pred_labels"],
                valid_log_output["gold_labels"],
                self.entity_types
            )
            logger.info("eval test")
            test_res_each_type = evaluate_each_entity_type(
                test_log_output["pred_labels"],
                test_log_output["gold_labels"],
                self.entity_types
            )
            logger.info("eval orig valid")
            orig_valid_res_each_type = evaluate_each_entity_type(
                valid_log_output["orig_pred_labels"],
                valid_log_output["gold_labels"],
                self.entity_types
            )
            logger.info("eval orig test")
            orig_test_res_each_type = evaluate_each_entity_type(
                test_log_output["orig_pred_labels"],
                test_log_output["gold_labels"],
                self.entity_types
            )


        log = {
            "epoch": epoch,
            "update": update,
            "alpha": alpha,
            "valid_loss": valid_loss,
            "valid_backbone_loss": valid_backbone_loss,
            "valid_ebpem_loss": valid_ebpem_loss,
            "valid_orig_be_loss": valid_orig_be_loss,
            "test_loss": test_loss,
            "test_backbone_loss": test_backbone_loss,
            "test_ebpem_loss": test_ebpem_loss,
            "test_orig_be_loss": test_orig_be_loss,
            "valid_p": valid_p,
            "valid_r": valid_r,
            "valid_f": valid_f,
            "test_p": test_p,
            "test_r": test_r,
            "test_f": test_f,
            "valid_orig_p": valid_orig_p,
            "valid_orig_r": valid_orig_r,
            "valid_orig_f": valid_orig_f,
            "test_orig_p": test_orig_p,
            "test_orig_r": test_orig_r,
            "test_orig_f": test_orig_f,
            "valid_res_each_type": valid_res_each_type,
            "test_res_each_type": test_res_each_type,
            "orig_valid_res_each_type": orig_valid_res_each_type,
            "orig_test_res_each_type": orig_test_res_each_type,
        }
        self.add_save_log(log)

    def add_save_log(self, log):
        self.save_log_list.append(log)

    def save_log(self):
        assert self.save_log_file[-1] != "/"
        save_log_dir = self.save_log_file[:self.save_log_file.rfind("/")]
        if not os.path.exists(save_log_dir):
            os.makedirs(save_log_dir)

        f = open(self.save_log_file, "wb")
        pkl.dump(self.save_log_list, f)
        f.close()