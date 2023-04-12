import json
import pandas

from detecter import parser
import os

def prepare_BCB_code() -> pandas.DataFrame:
    with open("dataset/BigCloneBench/data.jsonl.txt", "r") as f:
        jsonl = f.readlines()
    data_list = [json.loads(data) for data in jsonl]
    idx_list =[data["idx"] for data in data_list]
    code_list = [data["func"] for data in data_list]
    df = pandas.DataFrame({
        "index": idx_list,
        "code": code_list
	})
    df["cindex"] = df.index
    df = df.set_index("index")
    df["tree"] = df["code"].apply(lambda x: parser.parse(x, "java"))
    return df.loc[:, ["tree", "cindex"]]


def prepare_BCB_pair(path, code_data: pandas.DataFrame):
    with open(path, "r") as f:
        lines = f.readlines()
    pairs = [line.split() for line in lines]
    lhs_list = [str(pair[0]) for pair in pairs]
    rhs_list = [str(pair[1]) for pair in pairs]
    label_list = [bool(int(pair[2])) for pair in pairs]
    df = pandas.DataFrame({
        "lhs": lhs_list,
        "rhs": rhs_list,
        "label": label_list,
	})
    # df.loc[:, "lhs"] = code_data.loc[df["lhs"], "cindex"].values
    # df.loc[:, "rhs"] = code_data.loc[df["rhs"], "cindex"].values
    return df


def prepare_BCB_data():
    code_path = "dataset/BigCloneBench/data.jsonl.txt"
    if not os.path.exists(code_path + ".bin"):
        code_data = prepare_BCB_code()
        code_data.to_pickle(code_path + ".bin")
    else:
        code_data = pandas.read_pickle(code_path + ".bin")
    for mode in ["train", "valid", "test"]:
        path = "dataset/BigCloneBench/{}.txt".format(mode)
        if not os.path.exists(path + ".bin"):
            prepare_BCB_pair(path, code_data).to_pickle(path + ".bin")


if __name__ == "__main__":
    prepare_BCB_data()
