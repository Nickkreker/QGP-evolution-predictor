import torch
import numpy as np

path_to_models = 'models'

def predict(x, path_to_prediction, evolution_length=9):
    Ed = np.array([x])
    Vx = np.array([np.zeros_like(x)])
    Vy = np.array([np.zeros_like(x)])
    
    x = torch.unsqueeze(torch.unsqueeze(x, 0), 0)

    for t in range(9):
        model = torch.load(f'{path_to_models}/u{t}{t+1}.pth')
        model.eval()

        x = model(x)

        Ed = np.append(Ed, x[0, 0])
        Vx = np.append(Vx, x[0, 1])
        Vy = np.append(Vy, x[0, 2])

    return Ed, Vx, Vy

def save_evolution(evolution, path):
    short_prefix = 6 * ' '
    long_prefix = 8 * ' '
    
    Ed, Vx, Vy = evolution
    with open(f'{path}/snapshot_Ed.dat', 'w') as f:
        Ed0_str = '\n'.join(list(map(lambda x: short_prefix + np.array2string(x, separator=short_prefix)[1:-1], Ed[0])))
        Ed0_str = long_prefix + 0 + '\n' + Ed0_str + '\n'
        f.write(Ed0_str)
        for i in range(10):
            Ed_str = '\n'.join(list(map(lambda x: short_prefix + np.array2string(x, separator=short_prefix)[1:-1], Ed[i])))
            Ed_str = long_prefix + i + '\n' + Ed_str + '\n'
            f.write(Ed_str)

    with open(f'{path}/snapshot_Vx.dat', 'w') as f:
        Vx0_str = '\n'.join(list(map(lambda x: short_prefix + np.array2string(x, separator=short_prefix)[1:-1], Vx[0])))
        Vx0_str = long_prefix + 0 + '\n' + Vx0_str + '\n'
        f.write(Vx0_str)
        for i in range(10):
            Vx_str = '\n'.join(list(map(lambda x: short_prefix + np.array2string(x, separator=short_prefix)[1:-1], Vx[i])))
            Vx_str = long_prefix + i + '\n' + Vx_str + '\n'
            f.write(Vx_str)
    
    with open(f'{path}/snapshot_Vy.dat', 'w') as f:
        Vy0_str = '\n'.join(list(map(lambda x: short_prefix + np.array2string(x, separator=short_prefix)[1:-1], Vy[0])))
        Vy0_str = long_prefix + 0 + '\n' + Vy0_str + '\n'
        f.write(Vy0_str)
        for i in range(10):
            Vy_str = '\n'.join(list(map(lambda x: short_prefix + np.array2string(x, separator=short_prefix)[1:-1], Vy[i])))
            Vy_str = long_prefix + i + '\n' + Vy_str + '\n'
            f.write(Vx_str)
