from graph import process_sdat, process_adj_matrix

def process(dataset):

    process_sdat(f"./datasets/{dataset}/train.raw")
    process_sdat(f"./datasets/{dataset}/test.raw")

    process_adj_matrix(f"./datasets/{dataset}/train.raw")
    process_adj_matrix(f"./datasets/{dataset}/test.raw")

if __name__ == "__main__":
    process("headlines")