from accelerate.logging import get_logger
from launcher_config import parse_config_path
from trainer.simple_resnabladbtrainer import SimpleResNablaDBTrainer

logger = get_logger(__name__)

def main():
    config_path = parse_config_path(
        default_config_path="config/simple_res-nabladb_sd_hps_usez0_basesnr.yaml",
        description="Run SimpleResNablaDB training.",
    )
    trainer = SimpleResNablaDBTrainer(config_path)
    trainer.setup()

    trainer.validate_training_config()

    sample_batch_size = trainer.config['sample']['batch_size']
    n_batch_per_epoch = trainer.config['sample']['num_batches_per_epoch']
    train_batch_size = trainer.config['train']['batch_size']
    n_accum_steps = trainer.config['train']['gradient_accumulation_steps']
    updates_per_epoch = (sample_batch_size * n_batch_per_epoch) // (train_batch_size * n_accum_steps)

    first_epoch = trainer.load_from_checkpoint()
    global_step = first_epoch * updates_per_epoch

    trainer.train(first_epoch, global_step)


if __name__ == "__main__":
    main()
