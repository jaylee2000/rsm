from ml_collections import config_dict

def get_default_configs():
    config = config_dict.ConfigDict()


    training = config.training = config_dict.ConfigDict()
    training.lr = 5e-4
    training.adam_beta1 = 0.9
    training.adam_beta2 = 0.999
    training.adam_weight_decay = 1e-2
    training.adam_epsilon = 1.e-8
    training.max_grad_norm = 1.0
    training.num_inner_epochs = 1
    training.batch_size = 4 # 1 for value_net True
    training.gradient_accumulation_steps = 2 # 8 for value_net True
    training.num_epochs = 201
    training.mixed_precision = "bf16"
    training.allow_tf32 = True
    training.gradscaler_growth_interval = 2000
    training.coeff_terminal = 1000.0
    training.reward_masking = False
    training.use_jvp = False
    training.detach_dir = True
    training.quantile_clipping = True
    training.use_saved_rgrad_x1 = False


    model = config.model = config_dict.ConfigDict()
    model.lora_rank = 8
    model.reward_scale = 3e7
    model.timestep_fraction = 1.0
    model.value_layers_per_block = 1
    model.value_channel_width = (64, 128, 256, 256)
    model.unet_reg_scale = 0.0
    model.use_value_net = False
    model.eta_mode = 'quad'
    model.value_net_param = 'lora'

    experiment = config.experiment = config_dict.ConfigDict()
    experiment.method = 'Nabla-DB'


    sampling = config.sampling = config_dict.ConfigDict()
    sampling.num_steps = 20
    sampling.guidance_scale = 5.0
    sampling.batch_size = 8
    sampling.num_batches_per_epoch = 1
    sampling.low_var_subsampling = True


    pretrained = config.pretrained = config_dict.ConfigDict()
    pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    pretrained.revision = "main"
    pretrained.autoencodertiny = True


    logging = config.logging = config_dict.ConfigDict()
    logging.use_wandb = True
    logging.save_freq = 5
    logging.num_checkpoint_limit = 500
    logging.save_json = True
    logging.wandb_dir = './wandb/'
    logging.wandb_key = 'YOUR_WANDB_API_KEY'
    logging.proj_name = 'vggflow'

    saving = config.saving = config_dict.ConfigDict()
    saving.output_dir = './saved_models/'

    eval = config.eval = config_dict.ConfigDict()
    eval.enabled = False
    eval.every_epochs = 1
    eval.prompt_fn = ""
    eval.prompt_fn_kwargs = config_dict.ConfigDict()



    return config
