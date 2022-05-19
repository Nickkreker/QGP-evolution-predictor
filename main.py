import click
import time
from utils import predict, predict_single_model, save_evolution, read_init, plot_evolution

@click.command()
@click.argument('input', type=click.Path(exists=True))
@click.argument('output')
@click.option('--visualize', is_flag=True,
             help='Add visualisation of evolution to output.')
@click.option('--verbose', '-v', is_flag=True,
             help='Print additional runtime information.')
@click.option('--threshold', default=0.005,
             help='Values of Ed smaller than threshold are represented as zero. By default is 0.005.')
@click.option('--single', '-s', is_flag=True,
             help='If set uses single model to construct evolution')
def cli(input, output, visualize, verbose, threshold, single):
    start_time = time.perf_counter()
    if verbose:
        print('Reading input')
    
    Ed = read_init(input)

    if verbose:
        print('Calculating evolution')

    evolution = None
    if single:
        evolution = predict_single_model(Ed, threshold)
    else:
        evolution = predict(Ed, threshold)

    if verbose:
        print('Saving evolution to file')
    save_evolution(evolution, output)

    if visualize:
        plot_evolution(evolution, output)

    elapsed_time = time.perf_counter() - start_time
    if verbose:
        print(f'Runtime={elapsed_time}s')


