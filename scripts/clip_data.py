# 注意路径要自己稍微修改



def create_cliped_file(fp):
    f = open(fp,'r',encoding='utf-8')
    fp_out = fp + '.clip'
    f_out = open(fp_out,'w',encoding='utf-8')
    now_example_len = 0
    # cliped_corpus = [[]]
    # now_example = cliped_corpus[0]

    lines = f.readlines()
    last_line_split = ['','']
    num_sents = 0
    for line in lines:
        line_split = line.strip().split()

        if len(line_split) != 0:
            print(line,end='',file=f_out)
        now_example_len += 1
        if len(line_split) == 0 or \
                (line_split[0] in ['。','！','？']
                 and line_split[1] == 'O' and now_example_len>210):
            print('',file=f_out)
            now_example_len = 0
            num_sents += 1
        elif ((line_split[0] in ['，','；'] or (now_example_len>1 and last_line_split[0] == '…' and line_split[0] == '…'))
                 and line_split[1] == 'O' and now_example_len>210):
            print('',file=f_out)
            now_example_len = 0
            num_sents += 1

        elif line_split[1][0].lower() == 'e' and now_example_len>210:
            print('',file=f_out)
            now_example_len = 0
            num_sents += 1

        last_line_split = line_split
    print("number of sentences:", num_sents)
    f_out.close()

    f_check = open(fp_out,'r',encoding='utf-8')
    lines = f_check.readlines()
    cliped_examples = [[]]
    now_example = cliped_examples[0]
    for line in lines:
        line_split = line.strip().split()
        if len(line_split) == 0:
            cliped_examples.append([])
            now_example = cliped_examples[-1]
        else:
            now_example.append(line.strip())

    check = 0
    for example in cliped_examples:
        if len(example)>200:
            print(len(example),''.join(map(lambda x:x.split(' ')[0],example)))
            check += 1

    if check == 0:
        print('没句子超过200的长度')
    else:
        print("num of check:", check)

if __name__ == "__main__":
    data_name = "resume"
    data_dir = f"/newNAS/Workspaces/NLPGroup/juncw/summer_dataset/reduce_emb_dataset/{data_name}"
    create_cliped_file(f'{data_dir}/train.txt')
    print("------------------------------------")
    create_cliped_file(f'{data_dir}/dev.txt')
    print("------------------------------------")
    create_cliped_file(f'{data_dir}/test.txt')