import numpy as np
import torch

from models import ARCHI_REGISTRY

def build_model(cfg):
    archi_name = cfg.archi
    model = ARCHI_REGISTRY[archi_name].build_model(cfg)

    return model

def calculate_ner_match(predict_lists, golden_lists, label_type="BIOES"):
    sent_num = len(golden_lists)
    golden_full = []
    predict_full = []
    right_full = []
    right_tag = 0
    all_tag = 0
    for idx in range(0, sent_num):
        # word_list = sentence_lists[idx]
        golden_list = golden_lists[idx]
        predict_list = predict_lists[idx]
        for idy in range(len(golden_list)):
            if golden_list[idy] == predict_list[idy]:
                right_tag += 1
        all_tag += len(golden_list)
        gold_matrix = get_ner_BIOES(golden_list)
        pred_matrix = get_ner_BIOES(predict_list)
        # print "gold", gold_matrix
        # print "pred", pred_matrix
        right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
        golden_full += gold_matrix
        predict_full += pred_matrix
        right_full += right_ner
    right_num = len(right_full)
    golden_num = len(golden_full)
    predict_num = len(predict_full)
    # if predict_num == 0:
    #     precision = -1
    # else:
    #     precision = (right_num + 0.0) / predict_num
    # if golden_num == 0:
    #     recall = -1
    # else:
    #     recall = (right_num + 0.0) / golden_num
    # if (precision == -1) or (recall == -1) or (precision + recall) <= 0.:
    #     f_measure = -1
    # else:
    #     f_measure = 2 * precision * recall / (precision + recall)
    # accuracy = (right_tag + 0.0) / all_tag
    # # print "Accuracy: ", right_tag,"/",all_tag,"=",accuracy
    # print
    # "gold_num = ", golden_num, " pred_num = ", predict_num, " right_num = ", right_num
    return predict_num, golden_num, right_num

def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string

def get_ner_BIOES(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    end_label = 'E-'
    single_label = 'S-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i - 1))
            whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
            index_tag = current_label.replace(begin_label, "", 1)

        elif single_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i - 1))
            whole_tag = current_label.replace(single_label, "", 1) + '[' + str(i)
            tag_list.append(whole_tag)
            whole_tag = ""
            index_tag = ""
        elif end_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i))
            whole_tag = ''
            index_tag = ''
        else:
            continue
    if (whole_tag != '') & (index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i] + ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    # print stand_matrix
    return stand_matrix

def calculate_prf(n_preds, n_golds, n_inters):
    p = (n_inters * 1.0 / n_preds) * 100 if n_preds != 0 else 0
    r = (n_inters * 1.0 / n_golds) * 100 if n_golds != 0 else 0
    f = 2.0 * p * r / (p + r) if (p + r) != 0 else 0
    return p, r, f

def get_certain_labels(label_lists, entity_type):
    data_size = len(label_lists)
    new_label_lists = [[] for _ in range(data_size)]
    for i in range(data_size):
        for label in label_lists[i]:
            if label == "O" or label[2:] != entity_type:
                new_label_lists[i].append("O")
            elif label[2:] == entity_type:
                new_label_lists[i].append(label)
    return new_label_lists

def calculate_length(label_lists):
    entity_len = []
    for labels in label_lists:
        seq_len = len(labels)

        for i in range(seq_len):
            if labels[i].startswith("B"):
                start = i

                j = start + 1
                while True:
                    if j >= seq_len:
                        end = j
                        break
                    if labels[j].startswith("E"):
                        end = j
                        break
                    else:
                        j += 1

                if end < seq_len:
                    entity_len.append(end - start + 1)
            if labels[i].startswith("S"):
                entity_len.append(1)
    return entity_len

def evaluate_each_entity_type(predict_lists, golden_lists, entity_types):
    res = {}
    entity_len = []
    for t in entity_types:
        cur_pred_lists = get_certain_labels(predict_lists, t)
        cur_gold_lists = get_certain_labels(golden_lists, t)

        n_preds, n_golds, n_inters = calculate_ner_match(cur_pred_lists, cur_gold_lists)
        p, r, f = calculate_prf(n_preds, n_golds, n_inters)
        res[t] = {}
        res[t]["p"] = p
        res[t]["r"] = r
        res[t]["f"] = f
        res[t]["n_golds"] = n_golds + 1e-13
        cur_entity_len = calculate_length(cur_gold_lists)
        entity_len.extend(cur_entity_len)
        res[t]["total_len"] = sum(cur_entity_len)
        res[t]["avg_len"] = res[t]["total_len"] / res[t]["n_golds"]
    res["total_num_entities"] = len(entity_len)
    res["total_len_entities"] = sum(entity_len)
    print("ent_len mean: {:.2f}".format(np.mean(entity_len)))
    print("ent_len std: {:.4f}".format(np.std(entity_len, ddof=1)))

    res["entity_types"] = entity_types

    entity_types.sort(key=lambda x: res[x]["avg_len"],)
    for t in entity_types:
        d = res[t]
        print(
            t,
            "ratio: {:.2f}".format(d["n_golds"] / res["total_num_entities"]),
            "avg_len: {:.2f}".format(d["avg_len"]),
            "p={:.2f} r={:.2f} f={:.2f}".format(d["p"], d["r"], d["f"]),
        )
    print("")

    return res


def calculate_distance(preds, golds, max_range):
    sent_len = len(preds)
    assert len(golds) == sent_len, print(len(golds), sent_len)

    def range_list(mid):

        res = [mid]
        for i, j in zip(list(range(mid+1, mid+max_range+1)), list(reversed(range(mid-max_range,mid)))):
            res.extend([i,j])
        return res

    dist = 0
    ntoken = 0
    for i in range(sent_len):
        if preds[i].startswith('O') or preds[i].startswith('I'):
           continue
        cur_range_list = range_list(i)

        for idx in cur_range_list:
            if idx >= 0 and idx <= sent_len - 1:
                if golds[idx][0] in ['B', 'E', 'S']:
                    dist += abs(idx - i)
                    if abs(idx - i) != 0:
                        ntoken += 1
                    break
        # elif preds[i].startswith('E'):
        #     for idx in cur_range_list:
        #         if idx >= 0 and idx <= sent_len - 1 and golds[idx].startswith('E'):
        #             dist += abs(idx - i)
        #             if abs(idx - i) != 0:
        #                 ntoken += 1
        #             break
        # elif preds[i].startswith('S'):
        #     for idx in cur_range_list:
        #         if idx >= 0 and idx <= sent_len - 1 and golds[idx].startswith('S'):
        #             dist += abs(idx - i)
        #             if abs(idx - i) != 0:
        #                 ntoken += 1
        #             break


    return dist, ntoken


def evaluate_edit_distance_between_preds_and_golds(predict_lists, golden_lists, max_range):
    data_size = len(predict_lists)
    assert len(golden_lists) == data_size, print(len(golden_lists), data_size)
    total_dist = 0
    total_ntoken = 0
    for i in range(data_size):
        cur_dist, ntoken = calculate_distance(predict_lists[i], golden_lists[i], max_range)
        total_dist += cur_dist
        total_ntoken += ntoken
    return total_dist, total_ntoken


def write_wrong_result(predict_lists, golden_lists, orig_data_file, output_data_file):

    fr = open(orig_data_file, 'r')
    fw = open(output_data_file, 'w')

    orig_sents = []
    cur_sent = []
    for line in fr.readlines():
        line = line.strip()
        if line == '':
            orig_sents.append(cur_sent)
            cur_sent = []
            continue
        token, label, seg = line.split(' ')
        cur_sent.append(token)

    assert len(orig_sents) == len(predict_lists) == len(golden_lists)

    n_wrong = 0
    for i in range(len(orig_sents)):
        seq_len = len(orig_sents[i])
        flag = False
        for j in range(seq_len):
            if predict_lists[i][j] != golden_lists[i][j]:
                flag = True
                break
        if flag:
            n_wrong += 1
            for j in range(seq_len):
                fw.write(f"{orig_sents[i][j]}({golden_lists[i][j]}) ")
            fw.write('\n')
            for j in range(seq_len):
                fw.write(f"{orig_sents[i][j]}({predict_lists[i][j]}) ")
            fw.write('\n')
            fw.write('\n')

    print(f"data size = {len(orig_sents)}, wrong size = {n_wrong}")

    fr.close()
    fw.close()


# def old_evaluate():
#     class Span:
#         def __init__(self, left, right, entity):
#             self.left = left
#             self.right = right
#             self.entity = entity
#
#         def __eq__(self, other):
#             flag = (self.left == other.left and
#                     self.right == other.right and
#                     self.entity == other.entity)
#             return flag
#
#         def __hash__(self):
#             return hash((self.left, self.right, self.entity))
#
#     def gather_spans(labels):
#         seq_len = len(labels)
#
#         spans = set()
#         for i in range(seq_len):
#             if labels[i].startswith("B"):
#                 start = i
#
#                 j = start + 1
#                 while True:
#                     if j >= seq_len:
#                         end = j
#                         break
#                     if labels[j].startswith("E"):
#                         end = j
#                         break
#                     else:
#                         j += 1
#
#                 if end < seq_len:
#                     spans.add(Span(start, end, labels[i][2:]))
#             if labels[i].startswith("S"):
#                 spans.add(Span(i, i, labels[i][2:]))
#         return spans
#
#     def calculate_ner_match(pred_labels, gold_labels):
#         batch_size = len(pred_labels)
#         assert len(pred_labels) == len(gold_labels)
#
#         n_preds, n_golds, n_inters = 0, 0, 0
#         for id in range(batch_size):
#             assert len(pred_labels[id]) == len(gold_labels[id])
#             pred_spans = gather_spans(pred_labels[id])
#             gold_spans = gather_spans(gold_labels[id])
#             n_preds += len(pred_spans)
#             n_golds += len(gold_spans)
#             n_inters += len(gold_spans.intersection(pred_spans))
#
#         return n_preds, n_golds, n_inters


def is_nan(tensor):
    return bool(torch.sum(torch.isnan(tensor)))

