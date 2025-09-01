"""Parse documents with AMR parser."""
import math
import os
import pickle
import amrlib
import argparse
from tqdm import tqdm


def get_data(work_dir):
    """
    Get sentences that need to be parsed.
    """
    with open(os.path.join(work_dir, "data_samples.pkl"), "rb") as f:
        data = pickle.load(f)

    text_list = set()
    for topic in data:
        for item in data[topic]:
            text_list.add(" ".join(item[1]))

    tokenized_data_dir = os.path.join(work_dir, "tokenized_data")
    if not os.path.exists(tokenized_data_dir):
        os.makedirs(tokenized_data_dir)
    with open(os.path.join(work_dir, "tokenized_data/sentences.txt"), "w") as f:
        for item in text_list:
            f.write(item + '\n')
    print("get sentences finish")


def map_input_output(in_dir, out_dir):
    """Map input paths to output paths."""
    in_paths, out_paths = [], []

    for fn in os.listdir(in_dir):
        if fn.endswith(".txt"):
            in_paths.append(os.path.join(in_dir, fn))
            out_paths.append(os.path.join(out_dir, fn))
    return in_paths, out_paths


def batch_parse_amrlib(docs, parser):
    """Parse documents in batch."""
    # GSII is much faster
    spans = []
    sents = []
    for doc in docs:
        spans.append((len(sents), len(doc)))
        sents.extend(doc)
    graphs = parser.parse_sents(sents)
    results = []
    for start, offset in spans:
        # results.append("\n".join(graphs[start:start+offset]))   # for gsii
        results.append("\n\n".join(graphs[start:start+offset]))   # for t5 and spring
    return results


def parse(model_dir, work_dir, batch_size=10, workers=1, worker_id=0, device=0):
    """Parse documents."""
    print("Parsing documents with amr parser")
    # parser = None
    parser = amrlib.load_stog_model(model_dir=model_dir, device=device)  # for gsii

    tokenized_dir = os.path.join(work_dir, "tokenized_data")
    amr_dir = os.path.join(work_dir, "amr_data")
    if not os.path.exists(amr_dir):
        os.makedirs(amr_dir)
    in_paths, out_paths = map_input_output(tokenized_dir, amr_dir)
    in_paths = [_ for idx, _ in enumerate(in_paths) if (idx % workers) == worker_id]
    out_paths = [_ for idx, _ in enumerate(out_paths) if (idx % workers) == worker_id]
    # Filter parsed docs
    process_in, process_out = [], []
    for fin, fout in zip(in_paths, out_paths):
        if not os.path.exists(fout):
            process_in.append(fin)
            process_out.append(fout)
    # Parse
    error_in_paths = []
    error_out_paths = []
    tot_num = len(process_in)
    success_num = 0
    with tqdm(total=tot_num) as pbar:
        for i in range(math.ceil(tot_num / batch_size)):
            start, end = i * batch_size, (i+1) * batch_size
            docs = []
            for fp in process_in[start:end]:
                with open(fp, "r") as f:
                    content = f.read().strip().splitlines()
                docs.append(content)
            try:
                results = batch_parse_amrlib(docs, parser)
                for idx, fp in enumerate(process_out[start:end]):
                    with open(fp, "w") as f:
                        f.write(results[idx])
                        success_num += 1
                pbar.update(batch_size)
            except (AttributeError, RuntimeError, IndexError, TypeError):
                # print("Parse error detected.")
                error_in_paths.extend(process_in[start:end])
                error_out_paths.extend(process_out[start:end])
    # Re-do error batches
    error_files = []
    for fp_i, fp_o in zip(error_in_paths, error_out_paths):
        with open(fp_i, "r") as f:
            content = f.read().splitlines()
        try:
            results = batch_parse_amrlib([content], parser)
            with open(fp_o, "w") as f:
                f.write(results[0])
                success_num += 1
        except (AttributeError, RuntimeError, IndexError, TypeError):
            error_files.append(fp_i)
    print(f"Totally {success_num} docs succeeded, {len(error_files)} failed.")
    print("\n" + "\n".join(error_files))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="Causal-TimeBank")
    args = parser.parse_args()

    project_path = os.path.abspath('..')
    print("Project path: {}".format(project_path))
    work_dir = os.path.join(project_path, 'data', args.dataset_name)
    model_dir = "/data/username/PLM/model_parse_xfm_bart_large-v0_1_0"

    get_data(work_dir)
    parse(model_dir, work_dir, batch_size=10, workers=1, worker_id=0, device=1)
