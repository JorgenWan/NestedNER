import re

class Instance:

    def __init__(self, chars, labels):
        self.chars = chars
        self.labels = labels # entities or O

    def __len__(self):
        return len(self.words)

class Instances:

    def __init__(self, file, inst_list, digit2zero=True):
        self.file = file

        self.inst_list = inst_list

        self.digit2zero = digit2zero
        self.data_size = len(inst_list)

    def __len__(self):
        return len(self.inst_list)

    def __getitem__(self, item):
        return self.inst_list[item]

    @staticmethod
    def load_data(file, digit2zero=False):
        with open(file, 'r', encoding='utf-8') as f:
            chars, labels = [], []
            inst_list = []
            for line in f.readlines():
                line = line.rstrip()
                if line == "":
                    inst_list.append(Instance(chars, labels))
                    chars = []
                    labels = []
                    continue
                char, label = line.split(' ')
                if digit2zero:
                    char = re.sub('\d', '0', char)  # replace digit with 0
                chars.append(char)
                labels.append(label)

        print(f"Sentences: {len(inst_list)}")
        instances = Instances(file, inst_list, digit2zero)

        return instances


def bmoes_to_bioes(file):
    f_in = open(file, "r", encoding="utf-8")
    f_out = open(file+".bioes", "w", encoding="utf-8")

    for line in f_in.readlines():
        line = line.strip()
        if line == "":
            f_out.write("\n")
            continue
        char, label = line.split()
        if label.startswith("M-"):
            label = "I-"+label[2:]
        f_out.write(f"{char} {label}\n")

    f_in.close()
    f_out.close()

def bio_to_bioes(file):
    insts = Instances.load_data(file, digit2zero=False)

    f_out = open(file+".bioes", "w", encoding="utf-8")
    for inst in insts:
        chars = inst.chars
        labels = inst.labels
        seq_len = len(labels)
        for i in range(seq_len):
            cur_label = labels[i]
            if i == seq_len - 1:
                if cur_label.startswith('B'):
                    labels[i] = "S-"+cur_label[2:]
                elif cur_label.startswith('I'):
                    labels[i] = "E-"+cur_label[2:]
            else:
                next_label = labels[i + 1]
                if cur_label.startswith('B'):
                    if next_label.startswith('B') or next_label.startswith('O'):
                        labels[i] = "S-"+cur_label[2:]
                elif cur_label.startswith('I'):
                    if next_label.startswith('B') or next_label.startswith('O'):
                        labels[i] = "E-"+cur_label[2:]
        for i in range(seq_len):
            f_out.write(f"{chars[i]} {labels[i]}\n")
        f_out.write("\n")
    f_out.close()

if __name__ == "__main__":
    resume_dir = "/newNAS/Workspaces/NLPGroup/juncw/summer_dataset/reduce_emb_dataset/resume"
    bmoes_to_bioes(f"{resume_dir}/train.char.bmes")
    bmoes_to_bioes(f"{resume_dir}/dev.char.bmes")
    bmoes_to_bioes(f"{resume_dir}/test.char.bmes")

    onto4_dir = "/newNAS/Workspaces/NLPGroup/juncw/summer_dataset/reduce_emb_dataset/ontonote4"
    bmoes_to_bioes(f"{onto4_dir}/train.char.bmes")
    bmoes_to_bioes(f"{onto4_dir}/dev.char.bmes")
    bmoes_to_bioes(f"{onto4_dir}/test.char.bmes")

    msra_dir = "/newNAS/Workspaces/NLPGroup/juncw/summer_dataset/reduce_emb_dataset/msra"
    bmoes_to_bioes(f"{msra_dir}/train.char.bmes")
    bmoes_to_bioes(f"{msra_dir}/dev.char.bmes")
    bmoes_to_bioes(f"{msra_dir}/test.char.bmes")

    # weibo2_dir = "/newNAS/Workspaces/NLPGroup/juncw/summer_dataset/reduce_emb_dataset/weibo2"
    # bio_to_bioes(f"{weibo2_dir}/train.txt")
    # bio_to_bioes(f"{weibo2_dir}/dev.txt")
    # bio_to_bioes(f"{weibo2_dir}/test.txt")

    # people_daily_dir = "/newNAS/Workspaces/NLPGroup/juncw/summer_dataset/reduce_emb_dataset/people_daily"
    # bio_to_bioes(f"{people_daily_dir}/train.txt")
    # bio_to_bioes(f"{people_daily_dir}/dev.txt")
    # bio_to_bioes(f"{people_daily_dir}/test.txt")

    # synthe_dir = "/newNAS/Workspaces/NLPGroup/juncw/summer_dataset/reduce_emb_dataset/synthe/raw"
    # bio_to_bioes(f"{synthe_dir}/train.txt")
    # bio_to_bioes(f"{synthe_dir}/dev.txt")
    # bio_to_bioes(f"{synthe_dir}/test.txt")