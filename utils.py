import torch
import numpy as np
import matplotlib.pyplot as plt

path_to_models = 'models'

def predict(x, evolution_length=9):
    Ed = np.array((x))
    Vx = np.array((np.zeros_like(x)))
    Vy = np.array((np.zeros_like(x)))
    x = torch.from_numpy(x)
    x = torch.unsqueeze(torch.unsqueeze(x, 0), 0)

    for t in range(9):
        model = torch.load(f'{path_to_models}/u{t}{t+1}.pth')
        model.eval()

        x = model(x) 
        prediction = x.detach().numpy() 

        Ed = np.append(Ed, prediction[0, 0])
        Vx = np.append(Vx, prediction[0, 1])
        Vy = np.append(Vy, prediction[0, 2])

    return Ed.reshape((-1, 256, 256)), Vx.reshape((-1, 256, 256)), Vy.reshape((-1, 256, 256))

def save_evolution(evolution, path):
    short_prefix = 6 * ' '
    long_prefix = 8 * ' '
    
    Ed, Vx, Vy = evolution
    with open(f'{path}/snapshot_Ed.dat', 'w') as f:
        Ed0_str = '\n'.join(list(map(lambda x: short_prefix + np.array2string(x, separator=short_prefix,
                                                                              floatmode='fixed')[1:-1], Ed[0])))
        Ed0_str = long_prefix + f'{0:.8f}' + '\n' + Ed0_str + '\n'
        f.write(Ed0_str)
        for i in range(10):
            Ed_str = '\n'.join(list(map(lambda x: short_prefix + np.array2string(x, separator=short_prefix,
                                                                                 floatmode='fixed')[1:-1], Ed[i])))
            Ed_str = long_prefix + f'{i:.8f}' + '\n' + Ed_str + '\n'
            f.write(Ed_str)

    with open(f'{path}/snapshot_Vx.dat', 'w') as f:
        Vx0_str = '\n'.join(list(map(lambda x: short_prefix + np.array2string(x, separator=short_prefix,
                                                                              floatmode='fixed')[1:-1], Vx[0])))
        Vx0_str = long_prefix + f'{0:.8f}' + '\n' + Vx0_str + '\n'
        f.write(Vx0_str)
        for i in range(10):
            Vx_str = '\n'.join(list(map(lambda x: short_prefix + np.array2string(x, separator=short_prefix,
                                                                                 floatmode='fixed')[1:-1], Vx[i])))
            Vx_str = long_prefix + f'{i:.8f}' + '\n' + Vx_str + '\n'
            f.write(Vx_str)
    
    with open(f'{path}/snapshot_Vy.dat', 'w') as f:
        Vy0_str = '\n'.join(list(map(lambda x: short_prefix + np.array2string(x, separator=short_prefix,
                                                                              floatmode='fixed')[1:-1], Vy[0])))
        Vy0_str = long_prefix + f'{0:.8f}' + '\n' + Vy0_str + '\n'
        f.write(Vy0_str)
        for i in range(10):
            Vy_str = '\n'.join(list(map(lambda x: short_prefix + np.array2string(x, separator=short_prefix,
                                                                                 floatmode='fixed')[1:-1], Vy[i])))
            Vy_str = long_prefix + f'{i:.8f}' + '\n' + Vy_str + '\n'
            f.write(Vx_str)

def read_init(path):
    Ed = np.array([], dtype=np.float32)
    with open(path, 'r') as f:
        for idx, line in enumerate(f):
            if idx == 262:
                break
            if idx > 0:
                t = np.fromstring(" ".join(line.split()), sep=' ', dtype=np.float32)
                Ed = np.hstack((Ed, t))

    return Ed.reshape(261,261)[3:-2, 3:-2]

def plot_evolution(evolution, output, t_freeze=0.18, eps=0.01):
    Ed, Vx, Vy = evolution
    fig = plt.figure(figsize=(30,9))
    for i in range(len(Ed)):
        fig.add_subplot(3, len(Ed), i + 1)
        plt.imshow(Ed[i], alpha=0.9)
        plt.imshow(np.abs(Ed[i]- t_freeze) < eps, alpha=0.1)
        fig.add_subplot(3, len(Ed), i + len(Ed) + 1)
        plt.imshow(Vx[i], alpha=0.9)
        fig.add_subplot(3, len(Ed), i + 2 * len(Ed) + 1)
        plt.imshow(Vy[i], alpha=0.9)
    plt.savefig(f'{output}/evolution.png')

