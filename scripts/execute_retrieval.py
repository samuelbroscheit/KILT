# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import shutil

import hydra
from omegaconf import OmegaConf

from kilt import retrieval

logger = logging.getLogger()

def setup_logger(logger):
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    log_formatter = logging.Formatter("[%(thread)s] %(asctime)s [%(levelname)s] %(name)s: %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(log_formatter)
    logger.addHandler(console)

setup_logger(logger)

@hydra.main(config_path="../kilt_internal/configs", config_name="retrieval")
def main(cfg):

    logger.info("loading {} ...".format(cfg.retriever))

    if cfg.retriever == "drqa":
        # DrQA tf-idf
        from kilt.retrievers import DrQA_tfidf

        if cfg.retriever_config:
            retriever = DrQA_tfidf.DrQA.from_config_file(
                cfg.retriever, cfg.retriever_config
            )
        else:
            retriever = DrQA_tfidf.DrQA.from_default_config(cfg.retriever)
    elif cfg.retriever == "dpr":
        # DPR
        from kilt.retrievers import DPR_connector

        if cfg.retriever_config:
            retriever = DPR_connector.DPR.from_config_file(
                cfg.retriever, cfg.retriever_config
            )
        else:
            retriever = DPR_connector.DPR.from_default_config(cfg.retriever)
    elif cfg.retriever == "dpr_distr":
        # DPR distributed
        from kilt_internal.retrievers import DPR_distr_connector

        if cfg.retriever_config:
            retriever = DPR_distr_connector.DPR.from_config_file(
                cfg.retriever, cfg.retriever_config
            )
        else:
            raise Exception("No default configuration for DPR distributed!")

    elif cfg.retriever == "dpr_wafer":
        from kilt_internal.retrievers import DPR_Wafer_connector

        if cfg.retriever_config:
            # for overriding the retriever's config cfg.checkpoint_dir
            if hasattr(cfg, "checkpoint_dir"):
                kwargs = {"checkpoint_dir": cfg.checkpoint_dir}
            else:
                kwargs = {}
            retriever = DPR_Wafer_connector.DPR.from_config_file(
                cfg.retriever, cfg.retriever_config, **kwargs,
            )
            if cfg.output_folder is None or len(cfg.output_folder) == 0:
                cfg.output_folder = f"{retriever.cfg.checkpoint_dir}/predictions/{retriever.cfg.checkpoint_load_suffix}__{retriever.cfg.ctx_src}"

        else:
            raise Exception("No default configuration for dpr_wafer!")

    elif cfg.retriever == "dpr_wafer_remote":
        from kilt_internal.retrievers import DPR_Wafer_RemoteIndex_connector

        if cfg.retriever_config:
            # for overriding the retriever's config cfg.checkpoint_dir
            if hasattr(cfg, "checkpoint_dir"):
                kwargs = {"checkpoint_dir": cfg.checkpoint_dir}
            else:
                kwargs = {}
            retriever = DPR_Wafer_RemoteIndex_connector.DPR.from_config_file(
                cfg.retriever, cfg.retriever_config, **kwargs,
            )
            if cfg.output_folder is None or len(cfg.output_folder) == 0:
                cfg.output_folder = f"{retriever.cfg.checkpoint_dir}/predictions/{retriever.cfg.checkpoint_load_suffix}__{retriever.cfg.ctx_src}"

        else:
            raise Exception("No default configuration for dpr_wafer!")

    elif cfg.retriever == "blink":
        # BLINK
        from kilt.retrievers import BLINK_connector

        if cfg.retriever_config:
            retriever = BLINK_connector.BLINK.from_config_file(
                cfg.retriever, cfg.retriever_config
            )
        else:
            retriever = BLINK_connector.BLINK.from_default_config(cfg.retriever)
    elif cfg.retriever == "bm25":
        # BM25
        from kilt.retrievers import BM25_connector

        if cfg.retriever_config:
            retriever = BM25_connector.BM25.from_config_file(
                cfg.retriever, cfg.retriever_config
            )
        else:
            retriever = BM25_connector.BM25.from_default_config(cfg.retriever)
    elif cfg.retriever == "bm25_wafer":
        # BM25
        from kilt_internal.retrievers import BM25_Wafer_connector

        if cfg.retriever_config:
            retriever = BM25_Wafer_connector.BM25.from_config_file(
                cfg.retriever, cfg.retriever_config
            )
        else:
            retriever = BM25_Wafer_connector.BM25.from_default_config(cfg.retriever)
    else:
        raise ValueError("unknown retriever model")

    if cfg.output_folder is None or len(cfg.output_folder) == 0:
        cfg.output_folder = "./"

    output_folder = cfg.output_folder

    suffix = ""
    trial = 0
    while True:
        if os.path.exists(f"{output_folder}{suffix}"):
            suffix = f"__rerun_{trial}"
            trial += 1
        else:
            output_folder = f"{output_folder}{suffix}"
            break

    os.makedirs(output_folder, exist_ok=True)
    OmegaConf.save(retriever.cfg, f"{output_folder}/retriever_cfg.yaml")
    if hasattr(retriever, "checkpoint_cfg"):
        OmegaConf.save(retriever.checkpoint_cfg, f"{output_folder}/checkpoint_cfg.yaml")

    retrieval.run(
        test_config=cfg,
        ranker=retriever,
        logger=logger,
        output_folder=output_folder,
    )


if __name__ == "__main__":
    main()
