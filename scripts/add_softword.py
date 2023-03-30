import jieba
from tqdm import tqdm
dict_file = "/newNAS/Workspaces/NLPGroup/juncw/summer_dataset/" \
            "reduce_emb_dataset/pretrain_embed/ctb.tok.txt"
jieba.load_userdict(dict_file)

def f(chars, labels, f_out):
    seq_len = len(labels)
    assert len(chars) == len(labels)

    for i in range(seq_len): # this is special for weibo, there are many dirty words!
        char_i = chars[i]
        if len(char_i) > 1:
            print(char_i)
            chars[i] = char_i[:1]

    sentence = "".join(chars)
    words = list(jieba.cut(sentence))

    len_words = sum([len(word) for word in words])
    if len_words != seq_len:
        for word in words:
            print(word, len(word))
        for char in chars:
            print(char, len(char))
        exit(0)
    # assert len_words == seq_len, f"{len_words} -- {seq_len} -- {words} -- {chars}"

    idx = 0
    for word in words:
        if len(word) == 1:
            f_out.write(f"{chars[idx]} {labels[idx]} S\n")
            idx += 1
        else:
            len_word = len(word)
            f_out.write(f"{chars[idx]} {labels[idx]} B\n")
            idx += 1
            for i in range(len_word - 2):
                f_out.write(f"{chars[idx]} {labels[idx]} I\n")
                idx += 1
            f_out.write(f"{chars[idx]} {labels[idx]} E\n")
            idx += 1


def add_softword(file):
    f_in = open(file, "r", encoding="utf-8")
    f_out = open(file+".softword", "w", encoding="utf-8")

    cur_chars = []
    cur_labels = []

    for line in tqdm(f_in.readlines()):
        line = line.strip()
        if line == "":
            f(cur_chars, cur_labels, f_out)
            cur_chars = []
            cur_labels = []
            f_out.write("\n")
            continue
        char, label = line.split()
        cur_chars.append(char)
        cur_labels.append(label)

    f_in.close()
    f_out.close()

if __name__ == "__main__":
    resume_dir = "/newNAS/Workspaces/NLPGroup/juncw/summer_dataset/reduce_emb_dataset/resume"
    add_softword(f"{resume_dir}/train.txt")
    add_softword(f"{resume_dir}/dev.txt")
    add_softword(f"{resume_dir}/test.txt")

    onto4_dir = "/newNAS/Workspaces/NLPGroup/juncw/summer_dataset/reduce_emb_dataset/ontonote4"
    add_softword(f"{onto4_dir}/train.txt")
    add_softword(f"{onto4_dir}/dev.txt")
    add_softword(f"{onto4_dir}/test.txt")

    msra_dir = "/newNAS/Workspaces/NLPGroup/juncw/summer_dataset/reduce_emb_dataset/msra"
    add_softword(f"{msra_dir}/train.txt")
    add_softword(f"{msra_dir}/dev.txt")
    add_softword(f"{msra_dir}/test.txt")

    # weibo_dir = "/newNAS/Workspaces/NLPGroup/juncw/summer_dataset/reduce_emb_dataset/weibo"
    # add_softword(f"{weibo_dir}/train.txt")
    # add_softword(f"{weibo_dir}/dev.txt")
    # add_softword(f"{weibo_dir}/test.txt")

    # people_daily_dir = "/newNAS/Workspaces/NLPGroup/juncw/summer_dataset/reduce_emb_dataset/people_daily"
    # add_softword(f"{people_daily_dir}/train.txt")
    # add_softword(f"{people_daily_dir}/dev.txt")
    # add_softword(f"{people_daily_dir}/test.txt")

    # synthe_dir = "/newNAS/Workspaces/NLPGroup/juncw/summer_dataset/reduce_emb_dataset/synthe/raw"
    # add_softword(f"{synthe_dir}/train.txt.bioes")
    # add_softword(f"{synthe_dir}/dev.txt.bioes")
    # add_softword(f"{synthe_dir}/test.txt.bioes")