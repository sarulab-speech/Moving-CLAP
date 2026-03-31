import csv

def parse_segments_csv(file_path):
    label_dict = {}
    
    for line in open(file_path, 'r', encoding='utf-8'):
        if line.startswith('#'):
            continue
        line = line.strip("\n")
        ytid, start_seconds, end_seconds, labels = line.split(", ")
        
        # キーの作成
        key = ytid + ":" + str(int(float(start_seconds)))
            
        # ラベルをリストとして保存（カンマ区切り）
        label_list = labels.strip('"').split(',')

        label_dict[key] = label_list

    return label_dict

label_data = parse_segments_csv("../audioset/unbalanced_train_segments.csv")

def get_tag(csv_filename):
    ret = dict()
    with open(csv_filename, 'r', encoding='utf-8', newline="\n") as f:
        reader = csv.DictReader(f)
        for row in reader:
            audiocap_id = row["audiocap_id"]
            youtube_id = row["youtube_id"]
            start_time = int(float(row["start_time"]))
            k = youtube_id + ":" + str(start_time)
            assert k in label_data, k

            ret[audiocap_id] = label_data[k]
    return ret

tag_dict = dict()
for fname in ["train", "val", "test"]:
    tag_info = get_tag(f"../fixed_audiocaps/{fname}.csv")
    n0 = len(tag_dict)
    tag_dict.update(tag_info)
    assert len(tag_dict) - n0 == len(tag_info)

with open('info.csv', mode='w') as f:
    for key, labels in tag_dict.items():
        labels_str = ','.join(labels)
        print(key+":"+labels_str, file=f)
