from metaflow import FlowSpec, step, kubernetes, secrets, environment, model, current

class TorchSingleNode(FlowSpec):

    @step
    def start(self):
        self.config = {
            "s3": {
                "url": "https://outerbounds-datasets.s3.us-west-2.amazonaws.com/mnist/mnist_data.zip",
                "local_data_path": "."
            },
            "wandb_project": "my-wandb-project",
            "training": {
                "lr": 0.001,
                "epochs": 3
            }
        }
        self.next(self.train)

    @model
    @kubernetes(image="public.ecr.aws/outerbounds/torch-hydra:latest")
    @secrets(sources=['wandb-api-key']) # set WANDB_API_KEY env var in this secret
    @environment(vars={"WANDB_PROJECT": "outerbounds-torchrun-demo", "WANDB_WATCH": "all"})
    @step
    def train(self):
        from mymodule import train
        from hydra.core.config_store import ConfigStore
        from hydra import compose, initialize
        from omegaconf import OmegaConf

        cs = ConfigStore.instance()
        cs.store(name="my_config", node=self.config)
        with initialize(version_base=None):
            cfg = compose(config_name="my_config")
            save_model_dir = train(cfg)
            current.model.save(save_model_dir)
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == '__main__':
    TorchSingleNode()
