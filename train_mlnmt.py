import argparse
import logging
import pprint

from mlnmt import train
from mcg.stream import (get_tr_stream_mSrc,
                        get_logprob_streams_mSrc)
import config as cfg

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--proto",
                        default="get_config_EsZh2En_mSrc",
                        help="Prototype config to use for model configuration")
    args = parser.parse_args()

    config = getattr(cfg, args.proto)()

    logger.info("Model options:\n{}".format(pprint.pformat(config)))
    train(config,
          get_tr_stream_mSrc(config),
          get_logprob_streams_mSrc(config))
