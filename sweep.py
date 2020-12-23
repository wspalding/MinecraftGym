import wandb
from train import train

sweep_config = {
    'method': 'bayes', #grid, random, bayes
    'metric': {
      'name': 'average total reward',
      'goal': 'maximize'   
    },
    'parameters': {
        'learning_rate': {
            'values': [0.01]
        },
        'epochs': {
            'values': [3000, 5000, 7500, 10000]
        },
        'batch_size': {
            'values': [1, 32, 64]
        },
        'training_epochs': {
            'values': [1, 10]
        },
        'loss_function': {
            'values': ['mse', 'huber']
        },
        # 'optimizer': {
        #     'values': ['adam', 'sgd']
        # }
    },
    'early_terminate': {
        'type': 'hyperband',
        's': 2,
        'eta': 3,
        'max_iter': 27
    }
  
}



if(__name__ == '__main__'):
    sweep_id = wandb.sweep(sweep_config, 
                            project="space_invaders_sweep_1")

    wandb.agent(sweep_id, train)