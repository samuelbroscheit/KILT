# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import glob
import json
import math
import os
import filelock

import hydra

from kilt import kilt_utils as utils
from kilt.retrievers.base_retriever import Retriever


def output_file_name(output_folder, dataset_file, output_suffix=""):
    if not output_suffix:
        output_suffix = ""
    basename = os.path.basename(dataset_file)
    output_file = os.path.join(output_folder, basename) + output_suffix
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    return output_file

def run(
    test_config,
    ranker: Retriever,
    logger,
    debug=False,
    output_folder="",
    num_shards=1,
    shard_id=0,
):

    for dataset in test_config.evaluation_datasets:
        dataset = hydra.utils.instantiate(test_config.datasets[dataset])

        logger.info("TASK: {}".format(dataset.task_family))
        logger.info("DATASET: {}".format(dataset.name))

        output_file = output_file_name(output_folder, dataset.file, test_config.output_suffix)
        if os.path.exists(output_file):
            logger.info(
                "Skip output file {} that already exists.".format(output_file)
            )
            continue

        raw_data = utils.load_data(dataset.file)
        validated_data = {}
        queries_data = ranker.get_queries_data()
        if queries_data:
            ranker_provided_queries_data = True
        else:
            queries_data = []
            ranker_provided_queries_data = False

        for element in raw_data:
            if dataset.validate_datapoint(element, logger=logger):
                element = dataset.transform_query(element, test_config.question_transform_type)
                if element["id"] in validated_data:
                    raise ValueError("ids are not unique in input data!")
                validated_data[element["id"]] = element
                if not ranker_provided_queries_data:
                    queries_data.append(
                        {"query": element["input"], "id": element["id"]}
                    )
        if debug:
            # just consider the top10 datapoints
            queries_data = queries_data[:10]
            print("query_data: {}", format(queries_data))

        if num_shards > 1:
            len_all_query_ctxts = len(queries_data)
            shard_size = math.ceil(len_all_query_ctxts / num_shards)
            start_idx = shard_id * shard_size
            end_idx = start_idx + shard_size
            queries_data = queries_data[start_idx:end_idx]
            logger.info(f"sharded query_ctxy size: {len(queries_data)}")
        else:
            logger.info(f"query_ctxy size: {len(queries_data)}")

        ranker.set_queries_data(queries_data)

        # get predictions
        provenance = ranker.run()

        if len(provenance) != len(queries_data):
            logger.warning(
                "different numbers of queries: {} and predictions: {}".format(
                    len(queries_data), len(provenance)
                )
            )

        # write prediction files
        if provenance:
            logger.info("writing prediction file to {}".format(output_file))

            predictions = []
            for query_id in provenance.keys():
                if query_id in validated_data:
                    element = validated_data[query_id]
                    new_output = [{"provenance": provenance[query_id]}]
                    # append the answers
                    if "output" in element:
                        for o in element["output"]:
                            if "answer" in o:
                                new_output.append({"answer": o["answer"]})
                    element["output"] = new_output
                    predictions.append(element)

            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok=True)

            if num_shards > 1:

                lock = filelock.FileLock(output_file_name(output_folder, dataset.file, ".lock"))
                with lock:
                    with open(output_file, "w+") as outfile:
                        for p in predictions:
                            json.dump(p, outfile)
                            outfile.write("\n")
                    output_files = glob.glob(output_file_name(output_folder, dataset.file, ".[0-9]*-[0-9]*"))
                    if len(output_files) == num_shards:
                        sorted(output_files)
                        with open(output_file_name(output_folder, dataset.file, ""), "w") as cat:
                            for output_file in output_files:
                                cat.writelines(open(output_file))

            else:

                with open(output_file, "w+") as outfile:
                    for p in predictions:
                        json.dump(p, outfile)
                        outfile.write("\n")
